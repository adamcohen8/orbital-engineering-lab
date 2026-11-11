"""Engineering diagnostics around deterministic Controller Bench runs.

The helpers in this module only analyze configured scenarios and simulator
evidence.  They never approximate or replace OEL dynamics, navigation,
flight-software, controller, actuator, or mission execution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
import textwrap
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from sim.controller_lab.models import (
    ControllerBenchConfig,
    ControllerBenchObjective,
    ControllerVariant,
)
from sim.flight_software.profiles import materialize_use_case_profile
from sim.utils.frames import eci_relative_to_ric_rect, ric_dcm_ir_from_rv

ENGINEERING_SCHEMA = "oel.controller_bench.engineering.v1"
EVIDENCE_MANIFEST_SCHEMA = "oel.controller_bench.evidence_manifest.v1"


def lint_materialized_scenario(
    root: dict[str, Any],
    *,
    suite: ControllerBenchConfig,
    variant: ControllerVariant,
    case_name: str,
    objectives: tuple[ControllerBenchObjective, ...],
) -> list[dict[str, Any]]:
    """Find static contradictions after applying a complete bench variant."""

    findings: list[dict[str, Any]] = []
    target_id = suite.controller_target.object_id
    objects = dict(root.get("objects", {}) or {})
    target = dict(objects.get(target_id, {}) or {})
    fsw = dict(target.get("flight_software", {}) or {})
    profile_id = str(fsw.get("profile", "") or "")
    if profile_id:
        selection = materialize_use_case_profile(
            profile_id,
            params=dict(fsw.get("params", {}) or {}),
            hardware_profile=fsw.get("hardware_profile"),
            task_period_s=fsw.get("task_period_s"),
        )
        fsw = selection.to_config()
    params = dict(fsw.get("params", {}) or {})
    stack_id = str(fsw.get("stack", "") or "")
    custom_stack = bool(str(fsw.get("module", "") or "") and str(fsw.get("class_name", "") or ""))
    mode = str(params.get("translation_mode", "") or "")
    control_law = str(params.get("control_law", "") or "")
    effective_control_law = control_law or ("reference_pd" if stack_id == "fsw.rpo_reference" else "")
    context = {"case": case_name, "variant": variant.name, "object_id": target_id}

    def add(code: str, severity: str, message: str, *, path: str = "", suggestion: str = "") -> None:
        findings.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "path": path,
                "suggestion": suggestion,
                **context,
            }
        )

    if str(target.get("kind", "satellite")) == "satellite" and suite.controller_target.slot == "flight_software":
        if not stack_id and not custom_stack:
            add(
                "missing_flight_software_stack",
                "error",
                "The materialized satellite variant does not select a catalog or custom complete flight-software stack.",
                path=f"objects.{target_id}.flight_software",
            )

    label = f"{variant.name} {variant.description}".lower()
    advertised_laws = {
        "lqr": "hcw_lqr",
        "curvilinear": "curvilinear_ric_pd",
        "rmoe": "rmoe",
    }
    for token, expected in advertised_laws.items():
        if token in label and effective_control_law and expected not in effective_control_law:
            add(
                "advertised_control_law_mismatch",
                "error",
                f"The variant label advertises {token!r}, but the materialized control_law is {effective_control_law!r}.",
                path=f"objects.{target_id}.flight_software.params.control_law",
                suggestion="Rename the variant or select the advertised complete stack/control law.",
            )

    target_relative = params.get("target_relative_state_ric_m")
    if mode == "ric_hold" and target_relative is None:
        add(
            "implicit_zero_ric_target",
            "warning",
            "RIC hold has no explicit target_relative_state_ric_m and therefore relies on the stack default.",
            path=f"objects.{target_id}.flight_software.params.target_relative_state_ric_m",
            suggestion="Declare the six-component RIC target explicitly in engineering units.",
        )
    reference_mode = str(params.get("reference_mode", params.get("attitude_reference_mode", "")) or "")
    allowed_reference_modes = {"quaternion", "nadir", "velocity", "sun", "target", "ric", "thrust", "replay"}
    if reference_mode and reference_mode not in allowed_reference_modes:
        add(
            "unsupported_attitude_reference_mode",
            "error",
            f"Attitude reference mode {reference_mode!r} is not supported.",
            path=f"objects.{target_id}.flight_software.params.reference_mode",
        )
    if target_relative is not None and (not isinstance(target_relative, (list, tuple)) or len(target_relative) != 6):
        add(
            "invalid_ric_target_shape",
            "error",
            "target_relative_state_ric_m must contain six components.",
            path=f"objects.{target_id}.flight_software.params.target_relative_state_ric_m",
        )

    simulator = dict(root.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    orbit_step = _positive_float(orbit.get("orbit_substep_s", simulator.get("dt_s")))
    attitude_step = _positive_float(attitude.get("attitude_substep_s"))
    if bool(attitude.get("enabled", False)) and orbit_step is not None and attitude_step is not None:
        if attitude_step > orbit_step + 1.0e-15:
            add(
                "attitude_step_exceeds_orbit_step",
                "error",
                f"Attitude substep {attitude_step:g} s exceeds orbit substep {orbit_step:g} s.",
                path="simulator.dynamics.attitude.attitude_substep_s",
            )

    task_period = _positive_float(fsw.get("task_period_s"))
    sensor_period = _positive_float(dict(target.get("knowledge", {}) or {}).get("refresh_rate_s"))
    for name, value, path in (
        ("flight-software task", task_period, f"objects.{target_id}.flight_software.task_period_s"),
        ("sensor refresh", sensor_period, f"objects.{target_id}.knowledge.refresh_rate_s"),
    ):
        raw_value = _get_dotted(root, path)
        if raw_value is not None and value is None:
            add("invalid_cadence", "error", f"The {name} cadence must be finite and positive.", path=path)

    for objective in objectives:
        max_inside = objective.max_time_inside_keepout_s
        keepout = objective.keepout_radius_km or objective.min_range_km
        max_final = objective.max_final_range_km or objective.max_final_relative_distance_km
        if keepout is not None and max_inside == 0.0 and max_final is not None and max_final < keepout:
            add(
                "objective_keepout_terminal_contradiction",
                "error",
                f"Objective {objective.name or objective.kind!r} requires final range <= {max_final:g} km "
                f"while forbidding entry inside a {keepout:g} km keep-out radius.",
                suggestion="Separate an approach corridor from a protected keep-out zone or revise the terminal box.",
            )
        if objective.expected_translation_mode and mode and objective.expected_translation_mode != mode:
            add(
                "objective_mode_mismatch",
                "error",
                f"Objective expects translation mode {objective.expected_translation_mode!r}, but the variant selects {mode!r}.",
            )
        if objective.expected_control_law and effective_control_law and objective.expected_control_law != effective_control_law:
            add(
                "objective_control_law_mismatch",
                "error",
                f"Objective expects control law {objective.expected_control_law!r}, but the variant selects {effective_control_law!r}.",
            )

    active_reciprocal: list[tuple[str, str]] = []
    for object_id, raw_object in objects.items():
        object_fsw = dict(dict(raw_object or {}).get("flight_software", {}) or {})
        object_params = dict(object_fsw.get("params", {}) or {})
        reference_id = str(object_params.get("reference_object_id", "") or "")
        if object_fsw.get("stack") not in {None, "", "fsw.passive"} and reference_id:
            active_reciprocal.append((str(object_id), reference_id))
    for object_id, reference_id in active_reciprocal:
        if (reference_id, object_id) in active_reciprocal and object_id < reference_id:
            add(
                "reciprocal_active_relative_control",
                "warning",
                f"Both {object_id!r} and {reference_id!r} actively control relative to each other; verify this is intentional.",
                suggestion="Use one passive reference vehicle for ordinary RPO conformance probes.",
            )

    for burn_path, burn in _scheduled_burns(root):
        delta_v = _vector_norm(burn.get("delta_v_m_s", burn.get("delta_v_eci_m_s")))
        duration = _positive_float(burn.get("duration_s"))
        max_accel = _positive_float(params.get("max_acceleration_m_s2"))
        if delta_v is not None and duration is not None and max_accel is not None and delta_v > max_accel * duration * 1.000001:
            add(
                "scheduled_burn_exceeds_declared_authority",
                "warning",
                f"Scheduled burn requests {delta_v:g} m/s in {duration:g} s, above the declared "
                f"{max_accel:g} m/s² controller authority.",
                path=burn_path,
                suggestion="Increase burn duration/authority or make clipping an explicit test expectation.",
            )
    return findings


def write_run_engineering_artifacts(
    *,
    payload: dict[str, Any],
    scenario_root: dict[str, Any],
    suite: ControllerBenchConfig,
    variant: ControllerVariant,
    case_name: str,
    objectives: tuple[ControllerBenchObjective, ...],
    run_outdir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write per-run engineering artifacts and return artifact/summary mappings."""

    settings = suite.engineering
    if not settings.enabled:
        return {}, {}
    engineering_dir = run_outdir / "engineering"
    engineering_dir.mkdir(parents=True, exist_ok=True)
    object_id = suite.controller_target.object_id
    reference_id = _reference_object_id(scenario_root, object_id, objectives)
    analyses: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}

    if settings.stack_conformance:
        analyses["stack_conformance"] = analyze_stack_conformance(payload, scenario_root, object_id)
    if settings.frame_sign_probe:
        analyses["frame_sign_probe"] = analyze_frame_sign_probe(payload, scenario_root, object_id, reference_id)
    if settings.cadence_analysis:
        analyses["cadence_analysis"] = analyze_cadences(payload, scenario_root, object_id)
    if settings.authority_analysis:
        analyses["authority_analysis"] = analyze_authority(payload, scenario_root, object_id)
    if settings.stability_analysis:
        analyses["stability_response"] = analyze_stability_response(
            payload,
            scenario_root,
            object_id,
            reference_id,
        )

    for name, analysis in analyses.items():
        path = engineering_dir / f"{name}.json"
        path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
        artifacts[f"engineering_{name}_json"] = str(path)

    if settings.stability_analysis:
        response_artifacts = _write_closed_loop_response_artifacts(
            payload,
            scenario_root,
            object_id,
            reference_id,
            engineering_dir,
        )
        artifacts.update(response_artifacts)

    if settings.aligned_timeline:
        timeline = build_aligned_flight_data_timeline(
            payload,
            object_id=object_id,
            reference_object_id=reference_id,
            max_rows=settings.timeline_max_rows,
        )
        timeline_json = engineering_dir / "aligned_flight_data_timeline.json"
        timeline_csv = engineering_dir / "aligned_flight_data_timeline.csv"
        timeline_json.write_text(json.dumps(timeline, indent=2, sort_keys=True), encoding="utf-8")
        _write_rows_csv(timeline_csv, list(timeline.get("rows", []) or []))
        artifacts["engineering_aligned_timeline_json"] = str(timeline_json)
        artifacts["engineering_aligned_timeline_csv"] = str(timeline_csv)
        analyses["aligned_timeline"] = {
            "row_count": len(list(timeline.get("rows", []) or [])),
            "source_sample_count": timeline.get("source_sample_count"),
            "downsampled": timeline.get("downsampled"),
        }

    summary = {
        "schema": ENGINEERING_SCHEMA,
        "object_id": object_id,
        "reference_object_id": reference_id,
        "case_name": case_name,
        "variant_name": variant.name,
        "analyses": analyses,
    }
    summary_path = engineering_dir / "engineering_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["engineering_summary_json"] = str(summary_path)
    return artifacts, summary


def analyze_stack_conformance(payload: dict[str, Any], root: dict[str, Any], object_id: str) -> dict[str, Any]:
    evidence = _evidence(payload, object_id)
    invocations = [dict(item or {}) for item in list(evidence.get("invocations", []) or [])]
    outputs = [dict(item or {}) for item in list(evidence.get("outputs", []) or [])]
    receipts = [dict(item or {}) for item in list(evidence.get("receipts", []) or [])]
    realizations = [dict(item or {}) for item in list(evidence.get("realizations", []) or [])]
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str, *, status: str = "required") -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "status": status, "detail": detail})

    invocation_ids = [int(item.get("invocation_id", -1)) for item in invocations]
    output_ids = [int(item.get("invocation_id", -1)) for item in outputs]
    check("invocation_output_bijection", invocation_ids == output_ids, f"invocations={len(invocations)}, outputs={len(outputs)}")
    times = [int(item.get("invocation_time_ns", -1)) for item in invocations]
    check("monotonic_invocation_time", all(b > a for a, b in zip(times, times[1:])), f"samples={len(times)}")
    input_causal = all(_clock_ns(item.get("delivery_time")) >= _clock_ns(item.get("source_time")) for item in list(evidence.get("input_events", []) or []))
    check("input_delivery_causality", input_causal, f"inputs={len(list(evidence.get('input_events', []) or []))}")

    commands = [dict(command or {}) for output in outputs for command in list(output.get("commands", []) or [])]
    command_keys = {_packet_key(item.get("command_id")) for item in commands}
    receipt_keys = {_packet_key(item.get("command_id")) for item in receipts}
    accepted_receipt_keys = {
        _packet_key(item.get("command_id"))
        for item in receipts
        if _command_disposition(item.get("disposition")) in {"accepted", "duplicate"}
    }
    realized_keys = {_packet_key(item.get("source_command_id")) for item in realizations if item.get("source_command_id")}
    latest_realization_end_ns = max(
        (int(item.get("interval_end_ns", -1) or -1) for item in realizations),
        default=-1,
    )
    latest_evidence_ns = max(latest_realization_end_ns, max(times, default=-1))
    realizable_keys = {
        _packet_key(item.get("command_id"))
        for item in commands
        if _clock_ns(dict(item.get("validity", {}) or {}).get("not_before"))
        < latest_evidence_ns
    }
    check("command_receipt_traceability", command_keys <= receipt_keys, f"commands={len(command_keys)}, receipts={len(receipt_keys)}")
    check(
        "command_receipt_acceptance",
        command_keys <= accepted_receipt_keys,
        f"commands={len(command_keys)}, accepted_receipts={len(accepted_receipt_keys)}",
    )
    check(
        "command_realization_traceability",
        realizable_keys <= realized_keys or not realizable_keys,
        f"realizable_commands={len(realizable_keys)}, realized_sources={len(realized_keys)}",
    )
    finite = True
    for item in commands:
        payload_row = dict(item.get("payload", {}) or {})
        finite = finite and _all_numeric_finite(payload_row)
    for item in realizations:
        finite = finite and all(
            _finite_vector(item.get(key), length=3)
            for key in ("requested_force_n", "requested_torque_n_m", "realized_force_n", "realized_torque_n_m")
        )
    check("finite_boundary_values", finite, "All command/realization numeric fields must be finite.")

    fsw = _object_fsw(root, object_id)
    params = dict(fsw.get("params", {}) or {})
    diagnostics = _diagnostic_rows(evidence)
    selected_modes = sorted({str(item.get("selected_mode")) for item in diagnostics if item.get("selected_mode") is not None})
    control_laws = sorted({str(item.get("control_law")) for item in diagnostics if item.get("control_law") is not None})
    expected_mode = str(params.get("translation_mode", "") or "")
    expected_law = str(params.get("control_law", "") or "")
    if expected_mode:
        check("selected_mode_matches_config", expected_mode in selected_modes, f"expected_observed={expected_mode}, actual={selected_modes}")
    if expected_law:
        check("control_law_matches_config", expected_law in control_laws, f"expected_observed={expected_law}, actual={control_laws}")

    required = [item for item in checks if item["status"] == "required"]
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "flight_software_stack_conformance",
        "object_id": object_id,
        "stack_id": invocations[0].get("stack_id") if invocations else fsw.get("stack"),
        "passed": bool(required) and all(bool(item["passed"]) for item in required),
        "checks": checks,
        "counts": {
            "invocations": len(invocations),
            "inputs": len(list(evidence.get("input_events", []) or [])),
            "commands": len(commands),
            "receipts": len(receipts),
            "realizations": len(realizations),
            "snapshots": len(list(evidence.get("snapshots", []) or [])),
        },
    }


def analyze_frame_sign_probe(
    payload: dict[str, Any],
    root: dict[str, Any],
    object_id: str,
    reference_object_id: str,
) -> dict[str, Any]:
    fsw = _object_fsw(root, object_id)
    params = dict(fsw.get("params", {}) or {})
    mode = str(params.get("translation_mode", "") or "")
    if mode != "ric_hold" or not reference_object_id:
        return {
            "schema": ENGINEERING_SCHEMA,
            "analysis": "frame_sign_probe",
            "status": "not_applicable",
            "reason": "The automatic response-direction probe currently applies to explicit RIC hold cases.",
        }
    control_law = str(params.get("control_law", "reference_pd") or "reference_pd")
    if control_law == "rmoe_if_then":
        return {
            "schema": ENGINEERING_SCHEMA,
            "analysis": "frame_sign_probe",
            "status": "not_applicable",
            "control_law": control_law,
            "reason": (
                "RMOE feedback regulates drift, centers, and oscillation amplitudes in a priority sequence; "
                "its first acceleration is not required to align with instantaneous Cartesian position error."
            ),
        }
    truth = _truth(payload, object_id)
    reference = _truth(payload, reference_object_id)
    thrust = np.asarray(dict(payload.get("applied_thrust_by_object", {}) or {}).get(object_id, []), dtype=float)
    if min(len(truth), len(reference), len(thrust)) == 0:
        return {"schema": ENGINEERING_SCHEMA, "analysis": "frame_sign_probe", "status": "insufficient_evidence"}
    target_m = np.asarray(params.get("target_relative_state_ric_m", np.zeros(6)), dtype=float).reshape(6)
    initial_ric = eci_relative_to_ric_rect(truth[0, :6], reference[0, :6])
    error_ric_km = target_m[:3] / 1.0e3 - initial_ric[:3]
    active = np.where(np.linalg.norm(thrust[:, :3], axis=1) > 1.0e-15)[0]
    if active.size == 0 or float(np.linalg.norm(error_ric_km)) <= 1.0e-12:
        return {
            "schema": ENGINEERING_SCHEMA,
            "analysis": "frame_sign_probe",
            "status": "not_excited",
            "initial_position_error_ric_m": (error_ric_km * 1.0e3).tolist(),
        }
    idx = int(active[0])
    ref_idx = min(idx, len(reference) - 1)
    c_ir = ric_dcm_ir_from_rv(reference[ref_idx, :3], reference[ref_idx, 3:6])
    accel_ric = c_ir.T @ thrust[idx, :3]
    denominator = float(np.linalg.norm(accel_ric) * np.linalg.norm(error_ric_km))
    cosine = float(np.dot(accel_ric, error_ric_km) / denominator) if denominator > 0.0 else float("nan")
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "frame_sign_probe",
        "status": "observed",
        "control_law": control_law,
        "sample_index": idx,
        "time_s": _time(payload, idx),
        "initial_position_error_ric_m": (error_ric_km * 1.0e3).tolist(),
        "first_active_acceleration_ric_m_s2": (accel_ric * 1.0e3).tolist(),
        "response_alignment_cosine": cosine,
        "directionally_correct": bool(math.isfinite(cosine) and cosine > 0.0),
        "interpretation": "Advisory first-response sign probe; coupled dynamics and feed-forward can affect individual axes.",
    }


def analyze_cadences(payload: dict[str, Any], root: dict[str, Any], object_id: str) -> dict[str, Any]:
    fsw = _object_fsw(root, object_id)
    simulator = dict(root.get("simulator", {}) or {})
    dynamics = dict(simulator.get("dynamics", {}) or {})
    orbit = dict(dynamics.get("orbit", {}) or {})
    attitude = dict(dynamics.get("attitude", {}) or {})
    evidence = _evidence(payload, object_id)
    invocation_times = np.asarray(
        [float(dict(item or {}).get("invocation_time_ns", 0)) * 1.0e-9 for item in list(evidence.get("invocations", []) or [])],
        dtype=float,
    )
    periods = np.diff(invocation_times)
    configured_task = _positive_float(fsw.get("task_period_s"))
    actual_median = float(np.median(periods)) if periods.size else None
    jitter = float(np.max(np.abs(periods - (configured_task or np.median(periods))))) if periods.size else None
    orbit_step = _positive_float(orbit.get("orbit_substep_s", simulator.get("dt_s")))
    attitude_step = _positive_float(attitude.get("attitude_substep_s"))
    duration = _positive_float(simulator.get("duration_s")) or 0.0
    expected_releases = None
    nominal_release_errors: list[float] = []
    if configured_task is not None:
        expected_releases = int(math.floor(duration / configured_task + 1.0e-10)) + 1
        expected_times = np.arange(expected_releases, dtype=float) * configured_task
        if invocation_times.size:
            nominal_release_errors = [float(np.min(np.abs(invocation_times - expected))) for expected in expected_times]
    max_nominal_error = max(nominal_release_errors, default=None)
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "multirate_cadence",
        "configured": {
            "simulator_step_s": _positive_float(simulator.get("dt_s")),
            "orbit_substep_s": orbit_step,
            "attitude_substep_s": attitude_step,
            "flight_software_task_period_s": configured_task,
            "sensor_refresh_rate_s": _positive_float(dict(dict(root.get("objects", {}) or {}).get(object_id, {}) or {}).get("knowledge", {}).get("refresh_rate_s")) if isinstance(dict(dict(root.get("objects", {}) or {}).get(object_id, {}) or {}).get("knowledge"), dict) else None,
        },
        "observed": {
            "invocation_count": int(invocation_times.size),
            "expected_nominal_invocation_count": expected_releases,
            "median_task_period_s": actual_median,
            "max_task_period_jitter_s": jitter,
            "minimum_task_period_s": float(np.min(periods)) if periods.size else None,
            "maximum_task_period_s": float(np.max(periods)) if periods.size else None,
            "max_nominal_release_timing_error_s": max_nominal_error,
            "missed_nominal_release_count": sum(
                error > max(1.0e-9, (configured_task or 1.0) * 1.0e-9) for error in nominal_release_errors
            ),
        },
        "checks": {
            "attitude_not_slower_than_orbit": (
                True if not bool(attitude.get("enabled", False)) or attitude_step is None or orbit_step is None else attitude_step <= orbit_step
            ),
            "task_release_coverage": expected_releases is None or int(invocation_times.size) >= expected_releases,
            "task_period_matches_configuration": configured_task is None or max_nominal_error is not None and max_nominal_error <= max(1.0e-9, configured_task * 1.0e-9),
        },
        "note": "OEL generates task releases as hard events; releases are not rounded to orbit or attitude output samples.",
    }


def analyze_authority(payload: dict[str, Any], root: dict[str, Any], object_id: str) -> dict[str, Any]:
    rows = [dict(item or {}) for item in list(_evidence(payload, object_id).get("realizations", []) or [])]
    first_saturated = next((row for row in rows if bool(row.get("saturated", False))), None)
    requested_impulse = 0.0
    realized_impulse = 0.0
    force_residuals: list[float] = []
    torque_residuals: list[float] = []
    device_extrema: dict[str, float] = {}
    for row in rows:
        requested_force = _vector(row.get("requested_force_n"))
        realized_force = _vector(row.get("realized_force_n"))
        requested_torque = _vector(row.get("requested_torque_n_m"))
        realized_torque = _vector(row.get("realized_torque_n_m"))
        dt_s = max(0.0, (float(row.get("interval_end_ns", 0)) - float(row.get("interval_start_ns", 0))) * 1.0e-9)
        if requested_force is not None and realized_force is not None:
            requested_impulse += float(np.linalg.norm(requested_force)) * dt_s
            realized_impulse += float(np.linalg.norm(realized_force)) * dt_s
            force_residuals.append(float(np.linalg.norm(requested_force - realized_force)))
        if requested_torque is not None and realized_torque is not None:
            torque_residuals.append(float(np.linalg.norm(requested_torque - realized_torque)))
        for field_raw in list(row.get("device_state", []) or []):
            field = dict(field_raw or {})
            value = field.get("value")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                name = f"{row.get('actuator_id', '')}.{field.get('name', '')}"
                device_extrema[name] = max(device_extrema.get(name, 0.0), abs(float(value)))
    fsw = _object_fsw(root, object_id)
    params = dict(fsw.get("params", {}) or {})
    truth = _truth(payload, object_id)
    mass = float(truth[0, 13]) if truth.ndim == 2 and truth.shape[0] and truth.shape[1] > 13 else None
    max_accel = _positive_float(params.get("max_acceleration_m_s2"))
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "actuator_authority_and_feasibility",
        "object_id": object_id,
        "realization_count": len(rows),
        "saturated_count": sum(bool(row.get("saturated", False)) for row in rows),
        "saturation_fraction": sum(bool(row.get("saturated", False)) for row in rows) / len(rows) if rows else 0.0,
        "first_saturation": (
            None
            if first_saturated is None
            else {
                "actuator_id": first_saturated.get("actuator_id"),
                "interval_start_s": float(first_saturated.get("interval_start_ns", 0) or 0) * 1.0e-9,
                "interval_end_s": float(first_saturated.get("interval_end_ns", 0) or 0) * 1.0e-9,
            }
        ),
        "requested_impulse_n_s": requested_impulse,
        "realized_impulse_n_s": realized_impulse,
        "impulse_delivery_fraction": realized_impulse / requested_impulse if requested_impulse > 0.0 else None,
        "max_force_realization_error_n": max(force_residuals, default=0.0),
        "rms_force_realization_error_n": float(np.sqrt(np.mean(np.square(force_residuals)))) if force_residuals else 0.0,
        "max_torque_realization_error_n_m": max(torque_residuals, default=0.0),
        "initial_mass_kg": mass,
        "declared_max_acceleration_m_s2": max_accel,
        "declared_force_equivalent_n": mass * max_accel if mass is not None and max_accel is not None else None,
        "device_state_absolute_extrema": device_extrema,
        "interpretation": "Authority evidence compares typed demands with the physical actuator realizations used by OEL dynamics.",
    }


def analyze_stability_response(
    payload: dict[str, Any],
    root: dict[str, Any],
    object_id: str,
    reference_object_id: str,
) -> dict[str, Any]:
    fsw = _object_fsw(root, object_id)
    params = dict(fsw.get("params", {}) or {})
    state: np.ndarray
    response_kind: str
    if reference_object_id:
        truth = _truth(payload, object_id)
        reference = _truth(payload, reference_object_id)
        n = min(len(truth), len(reference))
        rows = []
        for idx in range(n):
            try:
                rows.append(eci_relative_to_ric_rect(truth[idx, :6], reference[idx, :6]))
            except (ValueError, FloatingPointError):
                rows.append(np.full(6, np.nan))
        state = np.asarray(rows, dtype=float)
        target = np.asarray(params.get("target_relative_state_ric_m", np.zeros(6)), dtype=float).reshape(6) / 1.0e3
        state = state - target
        response_kind = "relative_ric_error"
    else:
        truth = _truth(payload, object_id)
        state = truth[:, 10:13] if truth.ndim == 2 and truth.shape[1] >= 13 else np.empty((0, 3))
        response_kind = "body_rate"
    finite_rows = state[np.all(np.isfinite(state), axis=1)] if state.ndim == 2 else np.empty((0, 0))
    if len(finite_rows) < 3:
        return {"schema": ENGINEERING_SCHEMA, "analysis": "stability_response", "status": "insufficient_evidence"}
    norms = np.linalg.norm(finite_rows, axis=1)
    time_values = np.asarray(payload.get("time_s", []), dtype=float).reshape(-1)[: len(finite_rows)]
    peak_index = int(np.argmax(norms))
    recovery_threshold = 0.1 * float(norms[peak_index])
    recovery_time_s = None
    if time_values.size == len(norms):
        for idx in range(peak_index, len(norms)):
            if np.all(norms[idx:] <= recovery_threshold):
                recovery_time_s = float(time_values[idx] - time_values[peak_index])
                break
    response = {
        "initial_error_norm": float(norms[0]),
        "final_error_norm": float(norms[-1]),
        "contraction_ratio": float(norms[-1] / norms[0]) if norms[0] > 0.0 else None,
        "peak_to_initial_ratio": float(np.max(norms) / norms[0]) if norms[0] > 0.0 else None,
        "coordinate_zero_crossings": [int(np.count_nonzero(np.diff(np.signbit(finite_rows[:, idx])))) for idx in range(finite_rows.shape[1])],
        "rms_error_norm": float(np.sqrt(np.mean(norms**2))),
        "integrated_error_norm": (
            float(np.trapezoid(norms, time_values)) if time_values.size == len(norms) and len(norms) > 1 else None
        ),
        "peak_sample_index": peak_index,
        "ten_percent_recovery_time_s": recovery_time_s,
    }
    identification: dict[str, Any] = {"status": "insufficient_excitation"}
    x = finite_rows[:-1]
    y = finite_rows[1:]
    if len(x) >= finite_rows.shape[1] + 2:
        design = np.column_stack((x, np.ones(len(x))))
        coeff, _residuals, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
        a_matrix = coeff[:-1, :].T
        eigenvalues = np.linalg.eigvals(a_matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        identification = {
            "status": "identified",
            "method": "trajectory_least_squares_affine_discrete",
            "sample_count": len(x),
            "regression_rank": int(rank),
            "state_dimension": int(finite_rows.shape[1]),
            "a_matrix": a_matrix.tolist(),
            "eigenvalues": [{"real": float(value.real), "imag": float(value.imag), "magnitude": float(abs(value))} for value in eigenvalues],
            "spectral_radius": spectral_radius,
            "locally_contracting_discrete_fit": spectral_radius < 1.0,
        }
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "stability_response",
        "status": "analyzed",
        "response_kind": response_kind,
        "response": response,
        "local_linear_identification": identification,
        "interpretation": "The identified matrix is an advisory local empirical fit to the real closed-loop run, not a replacement dynamics model or a global stability proof.",
    }


def build_aligned_flight_data_timeline(
    payload: dict[str, Any],
    *,
    object_id: str,
    reference_object_id: str = "",
    max_rows: int = 20000,
) -> dict[str, Any]:
    time_s = np.asarray(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = _truth(payload, object_id)
    reference = _truth(payload, reference_object_id) if reference_object_id else np.empty((0, 0))
    belief = np.asarray(dict(payload.get("belief_by_object", {}) or {}).get(object_id, []), dtype=float)
    desired_attitude = np.asarray(
        dict(payload.get("desired_attitude_by_object", {}) or {}).get(object_id, []),
        dtype=float,
    )
    thrust = np.asarray(dict(payload.get("applied_thrust_by_object", {}) or {}).get(object_id, []), dtype=float)
    torque = np.asarray(dict(payload.get("applied_torque_by_object", {}) or {}).get(object_id, []), dtype=float)
    n = min(len(time_s), len(truth))
    if n <= 0:
        return {"schema": ENGINEERING_SCHEMA, "rows": [], "source_sample_count": 0, "downsampled": False}
    indices = np.arange(n)
    if n > max_rows:
        indices = np.unique(np.linspace(0, n - 1, max_rows, dtype=int))
    evidence = _evidence(payload, object_id)
    inv_times = np.asarray([float(dict(item or {}).get("invocation_time_ns", 0)) * 1.0e-9 for item in list(evidence.get("invocations", []) or [])])
    input_events = [dict(item or {}) for item in list(evidence.get("input_events", []) or [])]
    input_times = np.asarray([_clock_ns(item.get("delivery_time")) * 1.0e-9 for item in input_events])
    measurement_event_indices = [
        idx for idx, event in enumerate(input_events) if str(event.get("kind", "")) == "measurement"
    ]
    measurement_times = np.asarray([input_times[idx] for idx in measurement_event_indices], dtype=float)
    outputs = [dict(item or {}) for item in list(evidence.get("outputs", []) or [])]
    realizations = [dict(item or {}) for item in list(evidence.get("realizations", []) or [])]
    realization_times = np.asarray([float(item.get("interval_start_ns", 0)) * 1.0e-9 for item in realizations])
    receipt_times = np.asarray(
        [_clock_ns(dict(item or {}).get("received_at")) * 1.0e-9 for item in list(evidence.get("receipts", []) or [])]
    )
    diagnostics = _diagnostic_rows(evidence)
    rows: list[dict[str, Any]] = []
    for index in indices:
        row: dict[str, Any] = {"sample_index": int(index), "time_s": float(time_s[index])}
        _put_vector(row, "truth_position_eci_km", truth[index, :3] if truth.shape[1] >= 3 else None, ("x", "y", "z"))
        _put_vector(row, "truth_velocity_eci_km_s", truth[index, 3:6] if truth.shape[1] >= 6 else None, ("x", "y", "z"))
        if belief.ndim == 2 and index < len(belief) and belief.shape[1] >= 6:
            _put_vector(row, "estimated_position_eci_km", belief[index, :3], ("x", "y", "z"))
            _put_vector(row, "estimated_velocity_eci_km_s", belief[index, 3:6], ("x", "y", "z"))
            row["navigation_position_error_m"] = float(np.linalg.norm(belief[index, :3] - truth[index, :3]) * 1.0e3)
            row["navigation_velocity_error_m_s"] = float(np.linalg.norm(belief[index, 3:6] - truth[index, 3:6]) * 1.0e3)
        if desired_attitude.ndim == 2 and index < len(desired_attitude) and desired_attitude.shape[1] >= 4:
            _put_quaternion(row, "desired_attitude_quat_bn", desired_attitude[index, :4])
        if reference_object_id and reference.ndim == 2 and index < len(reference) and reference.shape[1] >= 6:
            relative = eci_relative_to_ric_rect(truth[index, :6], reference[index, :6])
            _put_vector(row, "relative_position_ric_m", relative[:3] * 1.0e3, ("r", "i", "c"))
            _put_vector(row, "relative_velocity_ric_m_s", relative[3:] * 1.0e3, ("r", "i", "c"))
        if thrust.ndim == 2 and index < len(thrust) and thrust.shape[1] >= 3:
            _put_vector(row, "applied_acceleration_eci_m_s2", thrust[index, :3] * 1.0e3, ("x", "y", "z"))
        if torque.ndim == 2 and index < len(torque) and torque.shape[1] >= 3:
            _put_vector(row, "applied_torque_body_n_m", torque[index, :3], ("x", "y", "z"))
        left = float(time_s[index - 1]) if index > 0 else -float("inf")
        right = float(time_s[index])
        row["fsw_invocations_since_previous_sample"] = int(np.count_nonzero((inv_times > left) & (inv_times <= right)))
        interval_inputs = np.where((input_times > left) & (input_times <= right))[0]
        row["input_events_since_previous_sample"] = int(interval_inputs.size)
        row["measurement_events_since_previous_sample"] = sum(
            str(input_events[int(item)].get("kind", "")) == "measurement" for item in interval_inputs
        )
        row["actuator_receipts_since_previous_sample"] = int(np.count_nonzero((receipt_times > left) & (receipt_times <= right)))
        row["actuator_intervals_started_since_previous_sample"] = int(np.count_nonzero((realization_times > left) & (realization_times <= right)))
        diagnostic_index = int(np.searchsorted(inv_times, right, side="right") - 1) if inv_times.size else -1
        if 0 <= diagnostic_index < len(diagnostics):
            for key in (
                "executive_phase",
                "goal_id",
                "goal_type",
                "goal_state",
                "selected_mode",
                "control_law",
                "control_phase",
                "control_saturated",
                "position_error_m",
                "velocity_error_m_s",
                "constraint_violation_count",
                "safety_review_violation_count",
                "pointing_error_rad",
                "pointing_compliant",
            ):
                if key in diagnostics[diagnostic_index]:
                    row[key] = diagnostics[diagnostic_index][key]
            output = outputs[diagnostic_index] if diagnostic_index < len(outputs) else {}
            commands = [dict(item or {}) for item in list(output.get("commands", []) or [])]
            row["controller_command_count"] = len(commands)
            if commands:
                latest_command = commands[-1]
                command_payload = dict(latest_command.get("payload", {}) or {})
                row["controller_command_actuator_id"] = latest_command.get("actuator_id")
                row["controller_command_frame"] = _get_dotted(latest_command, "frame.name")
                row["controller_command_schema"] = command_payload.get("schema")
                _put_vector(row, "controller_requested_force_n", command_payload.get("force_n"), ("x", "y", "z"))
                _put_vector(row, "controller_requested_torque_n_m", command_payload.get("torque_n_m"), ("x", "y", "z"))
        measurement_cursor = int(np.searchsorted(measurement_times, right, side="right") - 1) if measurement_times.size else -1
        if 0 <= measurement_cursor < len(measurement_event_indices):
            measurement = dict(input_events[measurement_event_indices[measurement_cursor]].get("payload", {}) or {})
            measurement_payload = dict(measurement.get("payload", {}) or {})
            row["latest_sensor_id"] = measurement.get("sensor_id")
            row["latest_measurement_type"] = measurement.get("measurement_type")
            row["latest_measurement_frame"] = _get_dotted(measurement, "frame.name")
            _put_vector(row, "latest_measured_position_m", measurement_payload.get("position_m"), ("x", "y", "z"))
            _put_vector(row, "latest_measured_velocity_m_s", measurement_payload.get("velocity_m_s"), ("x", "y", "z"))
            _put_vector(row, "latest_measured_body_rate_rad_s", measurement_payload.get("angular_rate_rad_s"), ("x", "y", "z"))
        realization_index = int(np.searchsorted(realization_times, right, side="right") - 1) if realization_times.size else -1
        if 0 <= realization_index < len(realizations):
            realization = realizations[realization_index]
            row["latest_realized_actuator_id"] = realization.get("actuator_id")
            row["latest_realization_saturated"] = bool(realization.get("saturated", False))
            _put_vector(row, "latest_realized_force_n", realization.get("realized_force_n"), ("x", "y", "z"))
            _put_vector(row, "latest_realized_torque_n_m", realization.get("realized_torque_n_m"), ("x", "y", "z"))
            device_state = list(realization.get("device_state", []) or [])
            if device_state:
                row["latest_actuator_device_state_json"] = json.dumps(device_state, sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return {
        "schema": ENGINEERING_SCHEMA,
        "analysis": "aligned_flight_data_timeline",
        "object_id": object_id,
        "reference_object_id": reference_object_id,
        "source_sample_count": n,
        "downsampled": len(indices) < n,
        "boundary_event_counts": {
            "invocations": len(list(evidence.get("invocations", []) or [])),
            "input_events": len(list(evidence.get("input_events", []) or [])),
            "outputs": len(list(evidence.get("outputs", []) or [])),
            "receipts": len(list(evidence.get("receipts", []) or [])),
            "realizations": len(list(evidence.get("realizations", []) or [])),
        },
        "rows": rows,
    }


def write_failure_reduction(result: dict[str, Any], outdir: Path) -> dict[str, str]:
    failed = [dict(run or {}) for run in list(result.get("runs", []) or []) if not bool(dict(run or {}).get("passed", False))]
    passed = [dict(run or {}) for run in list(result.get("runs", []) or []) if bool(dict(run or {}).get("passed", False))]
    reductions: list[dict[str, Any]] = []
    for run in failed:
        nearest = min(passed, key=lambda candidate: _run_distance(run, candidate), default=None)
        analysis = dict(run.get("failure_analysis", {}) or {})
        reductions.append(
            {
                "variant_name": run.get("variant_name"),
                "case_name": run.get("case_name"),
                "failure_count": analysis.get("failure_count", len(list(analysis.get("failures", []) or []))),
                "first_failure": (list(analysis.get("failures", []) or []) or [None])[0],
                "first_divergence": _first_divergence(run),
                "nearest_passing_run": (
                    None
                    if nearest is None
                    else {
                        "variant_name": nearest.get("variant_name"),
                        "case_name": nearest.get("case_name"),
                        "distance": _run_distance(run, nearest),
                    }
                ),
                "minimal_reproducer": dict(run.get("artifacts", {}) or {}).get("engineering_reproducer_suite_yaml"),
            }
        )
    packet = {
        "schema": "oel.controller_bench.failure_reduction.v1",
        "suite_name": result.get("suite_name"),
        "failed_run_count": len(failed),
        "passing_run_count": len(passed),
        "reductions": reductions,
        "method": "One case and one complete-stack variant per reproducer; nearest pass is selected from observed suite outcomes.",
    }
    path = outdir / "controller_bench_failure_reduction.json"
    path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    return {"failure_reduction_json": str(path)}


def write_suite_engineering_exports(result: dict[str, Any], outdir: Path) -> dict[str, str]:
    """Export conformance, signed-axis, and cadence outcome matrices."""

    runs = [dict(run or {}) for run in list(result.get("runs", []) or [])]
    conformance_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    cadence_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        engineering = dict(run.get("engineering", {}) or {})
        conformance = dict(engineering.get("stack_conformance", {}) or {})
        frame = dict(engineering.get("frame_sign_probe", {}) or {})
        cadence = dict(engineering.get("cadence_analysis", {}) or {})
        if conformance:
            conformance_rows.append(
                {
                    "variant_name": run.get("variant_name"),
                    "case_name": run.get("case_name"),
                    "stack_id": conformance.get("stack_id"),
                    "passed": conformance.get("passed"),
                    "failed_checks": [
                        item for item in list(conformance.get("checks", []) or []) if not bool(dict(item or {}).get("passed", True))
                    ],
                    "counts": conformance.get("counts", {}),
                }
            )
        if frame:
            frame_rows.append(
                {
                    "variant_name": run.get("variant_name"),
                    "case_name": run.get("case_name"),
                    "status": frame.get("status"),
                    "control_law": frame.get("control_law"),
                    "reason": frame.get("reason"),
                    "initial_position_error_ric_m": frame.get("initial_position_error_ric_m"),
                    "first_active_acceleration_ric_m_s2": frame.get("first_active_acceleration_ric_m_s2"),
                    "response_alignment_cosine": frame.get("response_alignment_cosine"),
                    "directionally_correct": frame.get("directionally_correct"),
                }
            )
        configured = dict(cadence.get("configured", {}) or {})
        task_period = configured.get("flight_software_task_period_s")
        if isinstance(task_period, (int, float)):
            base_variant = re.sub(r"__task_period_s_[^_]+(?:_[^_]+)?$", "", str(run.get("variant_name", "")))
            cadence_groups.setdefault((base_variant, str(run.get("case_name", ""))), []).append(run)

    cadence_rows: list[dict[str, Any]] = []
    for (base_variant, case_name), members in sorted(cadence_groups.items()):
        if len(members) < 2:
            continue
        metric_names = set.intersection(*(set(dict(item.get("metrics", {}) or {})) for item in members))
        metric_spreads: dict[str, Any] = {}
        for metric_name in sorted(metric_names):
            values: list[float] = []
            for member in members:
                try:
                    value = float(dict(member.get("metrics", {}) or {})[metric_name])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            if len(values) >= 2:
                spread = max(values) - min(values)
                metric_spreads[metric_name] = {
                    "minimum": min(values),
                    "maximum": max(values),
                    "absolute_spread": spread,
                    "relative_spread": spread / max(max(abs(value) for value in values), 1.0e-15),
                }
        cadence_rows.append(
            {
                "base_variant": base_variant,
                "case_name": case_name,
                "members": [
                    {
                        "variant_name": member.get("variant_name"),
                        "task_period_s": _get_dotted(member, "engineering.cadence_analysis.configured.flight_software_task_period_s"),
                        "passed": member.get("passed"),
                    }
                    for member in members
                ],
                "metric_spreads": metric_spreads,
            }
        )

    exports = {
        "stack_conformance_matrix_json": (
            outdir / "controller_bench_stack_conformance_matrix.json",
            {
                "schema": ENGINEERING_SCHEMA,
                "analysis": "stack_conformance_matrix",
                "summary": {
                    "run_count": len(conformance_rows),
                    "passed_count": sum(bool(row.get("passed")) for row in conformance_rows),
                    "failed_count": sum(not bool(row.get("passed")) for row in conformance_rows),
                },
                "rows": conformance_rows,
            },
        ),
        "frame_sign_matrix_json": (
            outdir / "controller_bench_frame_sign_matrix.json",
            {
                "schema": ENGINEERING_SCHEMA,
                "analysis": "frame_sign_matrix",
                "summary": {
                    "run_count": len(frame_rows),
                    "observed_count": sum(row.get("status") == "observed" for row in frame_rows),
                    "directionally_correct_count": sum(row.get("directionally_correct") is True for row in frame_rows),
                    "directionally_inconsistent_count": sum(row.get("directionally_correct") is False for row in frame_rows),
                    "not_excited_count": sum(row.get("status") == "not_excited" for row in frame_rows),
                    "not_applicable_count": sum(row.get("status") == "not_applicable" for row in frame_rows),
                },
                "rows": frame_rows,
            },
        ),
        "cadence_consistency_matrix_json": (
            outdir / "controller_bench_cadence_consistency_matrix.json",
            {
                "schema": ENGINEERING_SCHEMA,
                "analysis": "cadence_consistency_matrix",
                "summary": {"comparison_group_count": len(cadence_rows)},
                "rows": cadence_rows,
            },
        ),
    }
    artifacts: dict[str, str] = {}
    for artifact_id, (path, payload) in exports.items():
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        artifacts[artifact_id] = str(path)
    return artifacts


def write_minimal_reproducer(
    *,
    suite: ControllerBenchConfig,
    variant: ControllerVariant,
    case_name: str,
    case_config_path: Path,
    run_outdir: Path,
) -> dict[str, str]:
    engineering_dir = run_outdir / "engineering"
    engineering_dir.mkdir(parents=True, exist_ok=True)
    case_copy = engineering_dir / "reproducer_case.yaml"
    case_copy.write_text(case_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    variant_entry: dict[str, Any] = {"name": variant.name, "description": variant.description}
    if variant.pointer:
        variant_entry["controller"] = deepcopy(variant.pointer)
    if variant.pointers:
        variant_entry["pointers"] = [
            {"object_id": item.object_id, "slot": item.slot, "controller": deepcopy(item.pointer)} for item in variant.pointers
        ]
    if variant.overrides:
        variant_entry["overrides"] = {item.path: deepcopy(item.value) for item in variant.overrides}
    suite_payload = {
        "suite_name": f"reproduce_{suite.suite_name}_{case_name}_{variant.name}",
        "description": "Controller Bench minimal reproducer generated from one observed failed run.",
        "output_dir": "reproducer_output",
        "controller_target": {"object_id": suite.controller_target.object_id, "slot": suite.controller_target.slot},
        "save_run_payloads": True,
        "disable_plots": True,
        "disable_animations": True,
        "engineering": {"enabled": True},
        "variants": [variant_entry],
        "cases": [{"name": case_name, "config_path": case_copy.name}],
    }
    suite_path = engineering_dir / "reproducer_suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")
    return {
        "engineering_reproducer_suite_yaml": str(suite_path),
        "engineering_reproducer_case_yaml": str(case_copy),
    }


def write_evidence_manifest(result: dict[str, Any], outdir: Path) -> dict[str, str]:
    source_config = Path(str(result.get("config_path", "") or ""))
    inputs: list[dict[str, Any]] = []
    for path in [source_config, *(Path(str(item.get("config_path", "") or "")) for item in list(result.get("cases", []) or []))]:
        if path.is_file():
            inputs.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    artifacts: list[dict[str, Any]] = []
    for artifact_id, raw_path in sorted(dict(result.get("artifacts", {}) or {}).items()):
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if (
            path.is_file()
            and path.name != "controller_bench_evidence_manifest.json"
            and artifact_id not in {"summary_json", "agent_evidence_packet_json"}
        ):
            artifacts.append({"artifact_id": artifact_id, "path": str(path.resolve()), "sha256": _sha256(path)})
    implementation_paths = (
        Path(__file__),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("variants.py"),
        Path(__file__).with_name("cases.py"),
        Path(__file__).with_name("criteria.py"),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("objectives.py"),
        Path(__file__).with_name("diagnostics.py"),
        Path(__file__).with_name("optimization.py"),
        Path(__file__).with_name("reporting.py"),
        Path(__file__).with_name("runner.py"),
        Path(__file__).resolve().parents[1] / "flight_software" / "reference_stacks.py",
        Path(__file__).resolve().parents[1] / "gnc" / "orbit_v2.py",
        Path(__file__).resolve().parents[1] / "dynamics" / "orbit" / "relative_linear.py",
        Path(__file__).resolve().parents[1] / "control" / "orbit" / "lqr.py",
        Path(__file__).resolve().parents[1] / "runtime" / "satellites" / "flight_software_runtime.py",
        Path(__file__).resolve().parents[1] / "runtime" / "satellites" / "factory.py",
        Path(__file__).resolve().parents[1] / "actuators" / "command_bus.py",
        Path(__file__).resolve().parents[1] / "actuators" / "physical.py",
        Path(__file__).resolve().parents[1] / "dynamics" / "orbit" / "propagator.py",
        Path(__file__).resolve().parents[1] / "dynamics" / "orbit" / "accelerations.py",
        Path(__file__).resolve().parents[1] / "rocket" / "guidance.py",
        Path(__file__).resolve().parents[1] / "rocket" / "engine.py",
        Path(__file__).resolve().parents[1] / "actuators" / "physical.py",
    )
    manifest = {
        "schema": EVIDENCE_MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "suite_name": result.get("suite_name"),
        "controller_target": result.get("controller_target"),
        "inputs": inputs,
        "implementation_dependencies": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in implementation_paths if path.is_file()
        ],
        "selected_stack_ids": sorted(
            {
                str(dict(variant or {}).get("pointer", {}).get("stack", "") or "")
                for variant in list(result.get("variants", []) or [])
                if str(dict(variant or {}).get("pointer", {}).get("stack", "") or "")
            }
        ),
        "run_outcomes": [
            {
                "case_name": run.get("case_name"),
                "variant_name": run.get("variant_name"),
                "passed": bool(run.get("passed", False)),
                "failed_criteria": run.get("failed_criteria", []),
                "engineering_conformance_passed": _get_dotted(run, "engineering.stack_conformance.passed"),
            }
            for run in list(result.get("runs", []) or [])
        ],
        "semantic_lint": result.get("semantic_lint", {}),
        "artifacts": artifacts,
        "limitations": [
            "The manifest proves provenance and configured outcome evidence, not flight qualification.",
            "Empirical stability identification is local to each simulated trajectory.",
            "Safety recovery behavior remains owned by each flight-software stack.",
        ],
    }
    path = outdir / "controller_bench_evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"evidence_manifest_json": str(path)}


def analyze_completed_controller_bench(output_dir: str | Path) -> dict[str, Any]:
    outdir = Path(output_dir).expanduser().resolve()
    summary_path = outdir / "controller_bench_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Controller Bench summary not found: {summary_path}")
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    result.setdefault("artifacts", {}).update(write_failure_reduction(result, outdir))
    result.setdefault("artifacts", {}).update(write_suite_engineering_exports(result, outdir))
    result.setdefault("artifacts", {}).update(write_evidence_manifest(result, outdir))
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return {
        "suite_name": result.get("suite_name"),
        "output_dir": str(outdir),
        "failed_run_count": sum(not bool(run.get("passed", False)) for run in list(result.get("runs", []) or [])),
        "artifacts": {
            key: value
            for key, value in dict(result.get("artifacts", {}) or {}).items()
            if key in {"failure_reduction_json", "evidence_manifest_json"}
        },
    }


def reduce_controller_bench_failure(
    config_path: str | Path,
    *,
    case_name: str,
    variant_name: str,
    parameter_path: str = "",
    iterations: int = 8,
) -> dict[str, Any]:
    """Bisect a scalable initial condition using real Controller Bench runs."""

    if iterations < 1 or iterations > 20:
        raise ValueError("Failure reduction iterations must be in [1, 20].")
    from sim.controller_lab.cases import validate_controller_bench_config
    from sim.controller_lab.config import load_controller_bench_config
    from sim.controller_lab.objectives import _run_single_bench_case_with_payload

    preflight = validate_controller_bench_config(config_path, compare_names=[variant_name])
    if not bool(preflight.get("passed", False)):
        raise ValueError("Failure-reduction preflight failed: " + "; ".join(list(preflight.get("errors", []) or [])))
    suite = load_controller_bench_config(config_path)
    case = next((item for item in suite.cases if item.name == case_name), None)
    variant = next((item for item in suite.variants if item.name == variant_name), None)
    if case is None:
        raise ValueError(f"Unknown reduction case {case_name!r}.")
    if variant is None:
        raise ValueError(f"Unknown reduction variant {variant_name!r}.")
    root = yaml.safe_load(case.config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(root, dict):
        raise ValueError("Failure-reduction case root must be a mapping.")
    if not parameter_path:
        candidate = f"objects.{suite.controller_target.object_id}.initial_state.relative_ric_rect"
        if _get_dotted(root, candidate) is None:
            raise ValueError("No default scalable relative_ric_rect was found; provide --path.")
        parameter_path = candidate
    initial_value = _get_dotted(root, parameter_path)
    try:
        vector = np.asarray(initial_value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reduction path {parameter_path!r} must resolve to numeric scalar/vector data.") from exc
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"Reduction path {parameter_path!r} must resolve to finite numeric data.")

    search_dir = suite.output_dir / "failure_reduction_search" / f"{case_name}__{variant_name}"
    search_dir.mkdir(parents=True, exist_ok=True)
    reduced_engineering = replace(suite.engineering, enabled=False)
    reduced_optimization = replace(suite.optimization, enabled=False)
    samples: list[dict[str, Any]] = []

    def evaluate(scale: float) -> dict[str, Any]:
        scaled_root = deepcopy(root)
        scaled_value = (vector * float(scale)).tolist() if vector.ndim else float(vector) * float(scale)
        _set_dotted(scaled_root, parameter_path, scaled_value)
        scale_name = f"scale_{scale:.10f}".replace(".", "p")
        generated_path = search_dir / f"{scale_name}.yaml"
        generated_path.write_text(yaml.safe_dump(scaled_root, sort_keys=False), encoding="utf-8")
        generated_case = replace(case, name=scale_name, config_path=generated_path)
        run_suite = replace(
            suite,
            output_dir=search_dir,
            cases=(generated_case,),
            variants=(variant,),
            parallel_enabled=False,
            optimization=reduced_optimization,
            engineering=reduced_engineering,
        )
        run, _payload = _run_single_bench_case_with_payload(run_suite, generated_case, variant)
        sample = {
            "scale": float(scale),
            "passed": bool(run.get("passed", False)),
            "metrics": run.get("metrics", {}),
            "failed_criteria": run.get("failed_criteria", []),
            "config_path": str(generated_path),
            "output_dir": dict(run.get("artifacts", {}) or {}).get("output_dir"),
        }
        samples.append(sample)
        return sample

    zero = evaluate(0.0)
    full = evaluate(1.0)
    status = "reduced"
    lower_passing = 0.0
    upper_failing = 1.0
    if not zero["passed"]:
        status = "fails_at_zero_scale"
        upper_failing = 0.0
    elif full["passed"]:
        status = "baseline_passes"
        lower_passing = 1.0
    else:
        for _ in range(iterations):
            midpoint = 0.5 * (lower_passing + upper_failing)
            sample = evaluate(midpoint)
            if sample["passed"]:
                lower_passing = midpoint
            else:
                upper_failing = midpoint

    smallest_root = deepcopy(root)
    smallest_value = (vector * upper_failing).tolist() if vector.ndim else float(vector) * upper_failing
    _set_dotted(smallest_root, parameter_path, smallest_value)
    smallest_path = search_dir / "smallest_observed_failing_case.yaml"
    smallest_path.write_text(yaml.safe_dump(smallest_root, sort_keys=False), encoding="utf-8")
    packet = {
        "schema": "oel.controller_bench.failure_search.v1",
        "suite_name": suite.suite_name,
        "case_name": case_name,
        "variant_name": variant_name,
        "parameter_path": parameter_path,
        "original_value": np.asarray(initial_value).tolist(),
        "status": status,
        "largest_observed_passing_scale": lower_passing,
        "smallest_observed_failing_scale": upper_failing,
        "smallest_observed_failing_value": smallest_value,
        "smallest_observed_failing_case": str(smallest_path),
        "samples": sorted(samples, key=lambda item: float(item["scale"])),
        "method": "Deterministic scalar bisection of the selected numeric initial-condition path.",
        "caveat": "The boundary is empirical and assumes locally monotonic pass/fail behavior over the sampled scale interval.",
    }
    packet_path = search_dir / "failure_reduction_search.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    packet["failure_reduction_search_json"] = str(packet_path)
    return packet


def _evidence(payload: dict[str, Any], object_id: str) -> dict[str, Any]:
    return dict(dict(payload.get("flight_software_evidence_by_object", {}) or {}).get(object_id, {}) or {})


def _truth(payload: dict[str, Any], object_id: str) -> np.ndarray:
    return np.asarray(dict(payload.get("truth_by_object", {}) or {}).get(object_id, []), dtype=float)


def _object_fsw(root: dict[str, Any], object_id: str) -> dict[str, Any]:
    return dict(dict(dict(root.get("objects", {}) or {}).get(object_id, {}) or {}).get("flight_software", {}) or {})


def _reference_object_id(
    root: dict[str, Any],
    object_id: str,
    objectives: tuple[ControllerBenchObjective, ...],
) -> str:
    params = dict(_object_fsw(root, object_id).get("params", {}) or {})
    configured = str(params.get("reference_object_id", "") or "")
    if configured:
        return configured
    for objective in objectives:
        if objective.object_id == object_id and objective.reference_object_id:
            return objective.reference_object_id
    target = dict(dict(root.get("objects", {}) or {}).get(object_id, {}) or {})
    relative_to = str(dict(target.get("initial_state", {}) or {}).get("relative_to", "") or "")
    if relative_to:
        return relative_to
    return ""


def _diagnostic_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_raw in list(evidence.get("outputs", []) or []):
        output = dict(output_raw or {})
        row: dict[str, Any] = {"invocation_id": output.get("invocation_id")}
        for telemetry_raw in list(output.get("telemetry", []) or []):
            for field_raw in list(dict(telemetry_raw or {}).get("fields", []) or []):
                field = dict(field_raw or {})
                if field.get("name") is not None:
                    row[str(field["name"])] = field.get("value")
        rows.append(row)
    return rows


def _clock_ns(value: Any) -> int:
    row = dict(value or {}) if isinstance(value, dict) else {}
    return int(row.get("ticks", 0) or 0) * int(row.get("tick_period_ns", 1) or 1)


def _packet_key(value: Any) -> tuple[str, str, int]:
    row = dict(value or {}) if isinstance(value, dict) else {}
    sequence = row.get("sequence")
    return (
        str(row.get("source_id", "") or ""),
        str(row.get("boot_id", "") or ""),
        int(sequence if sequence is not None else -1),
    )


def _command_disposition(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", 1)[-1].lower()


def _all_numeric_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numeric_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _finite_vector(value: Any, *, length: int) -> bool:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return array.size == length and bool(np.all(np.isfinite(array)))


def _vector(value: Any) -> np.ndarray | None:
    return np.asarray(value, dtype=float).reshape(3) if _finite_vector(value, length=3) else None


def _vector_norm(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return float(np.linalg.norm(array)) if array.size and np.all(np.isfinite(array)) else None


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _get_dotted(root: Any, path: str, default: Any = None) -> Any:
    current = root
    for token in str(path).split("."):
        if not isinstance(current, dict) or token not in current:
            return default
        current = current[token]
    return current


def _set_dotted(root: dict[str, Any], path: str, value: Any) -> None:
    tokens = [token for token in str(path).split(".") if token]
    if not tokens:
        raise ValueError("Parameter path must be non-empty.")
    current: Any = root
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(path)
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            current = current[int(token)]
        else:
            raise KeyError(path)
    final = tokens[-1]
    if isinstance(current, dict) and final in current:
        current[final] = deepcopy(value)
    elif isinstance(current, list) and final.isdigit():
        current[int(final)] = deepcopy(value)
    else:
        raise KeyError(path)


def _scheduled_burns(root: Any, prefix: str = "") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(root, dict):
        if any(key in root for key in ("delta_v_m_s", "delta_v_eci_m_s")) and "duration_s" in root:
            yield prefix or "<root>", root
        for key, value in root.items():
            yield from _scheduled_burns(value, f"{prefix}.{key}".strip("."))
    elif isinstance(root, list):
        for index, value in enumerate(root):
            yield from _scheduled_burns(value, f"{prefix}.{index}".strip("."))


def _put_vector(row: dict[str, Any], prefix: str, value: Any, axes: tuple[str, str, str]) -> None:
    if value is None:
        return
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3:
        return
    for axis, component in zip(axes, array[:3]):
        row[f"{prefix}_{axis}"] = float(component)


def _put_quaternion(row: dict[str, Any], prefix: str, value: Any) -> None:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 4:
        return
    for axis, component in zip(("w", "x", "y", "z"), array[:4]):
        row[f"{prefix}_{axis}"] = float(component)


def _time(payload: dict[str, Any], index: int) -> float | None:
    values = list(payload.get("time_s", []) or [])
    return float(values[index]) if 0 <= index < len(values) else None


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fieldnames:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_closed_loop_response_artifacts(
    payload: dict[str, Any],
    root: dict[str, Any],
    object_id: str,
    reference_object_id: str,
    engineering_dir: Path,
) -> dict[str, str]:
    time_s = np.asarray(payload.get("time_s", []), dtype=float).reshape(-1)
    truth = _truth(payload, object_id)
    rows: list[dict[str, Any]] = []
    if reference_object_id:
        reference = _truth(payload, reference_object_id)
        fsw = _object_fsw(root, object_id)
        target = np.asarray(dict(fsw.get("params", {}) or {}).get("target_relative_state_ric_m", np.zeros(6)), dtype=float) / 1.0e3
        n = min(len(time_s), len(truth), len(reference))
        for idx in range(n):
            relative = eci_relative_to_ric_rect(truth[idx, :6], reference[idx, :6]) - target
            rows.append(
                {
                    "time_s": float(time_s[idx]),
                    "position_error_norm_m": float(np.linalg.norm(relative[:3]) * 1.0e3),
                    "velocity_error_norm_m_s": float(np.linalg.norm(relative[3:]) * 1.0e3),
                }
            )
    elif truth.ndim == 2 and truth.shape[1] >= 13:
        n = min(len(time_s), len(truth))
        for idx in range(n):
            rows.append({"time_s": float(time_s[idx]), "body_rate_norm_rad_s": float(np.linalg.norm(truth[idx, 10:13]))})
    else:
        return {}
    thrust = np.asarray(dict(payload.get("applied_thrust_by_object", {}) or {}).get(object_id, []), dtype=float)
    torque = np.asarray(dict(payload.get("applied_torque_by_object", {}) or {}).get(object_id, []), dtype=float)
    for idx, row in enumerate(rows):
        if thrust.ndim == 2 and idx < len(thrust) and thrust.shape[1] >= 3:
            row["applied_acceleration_norm_m_s2"] = float(np.linalg.norm(thrust[idx, :3]) * 1.0e3)
        if torque.ndim == 2 and idx < len(torque) and torque.shape[1] >= 3:
            row["applied_torque_norm_n_m"] = float(np.linalg.norm(torque[idx, :3]))
    csv_path = engineering_dir / "closed_loop_response.csv"
    _write_rows_csv(csv_path, rows)
    return {"engineering_closed_loop_response_csv": str(csv_path)}


def render_closed_loop_response_plots(result: dict[str, Any]) -> None:
    """Render response plots serially after parallel simulator workers finish."""

    cache_root = Path(tempfile.gettempdir()) / "oel-matplotlib"
    mpl_config = cache_root / "config"
    xdg_cache = cache_root / "cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        result["engineering_plot_error"] = str(exc)
        return
    for run in list(result.get("runs", []) or []):
        artifacts = dict(run.get("artifacts", {}) or {})
        csv_path = Path(str(artifacts.get("engineering_closed_loop_response_csv", "") or ""))
        if not csv_path.is_file():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        response_labels = [
            label
            for label in ("position_error_norm_m", "velocity_error_norm_m_s", "body_rate_norm_rad_s")
            if label in rows[0]
        ]
        effort_labels = [
            label
            for label in ("applied_acceleration_norm_m_s2", "applied_torque_norm_n_m")
            if label in rows[0]
        ]
        display_labels = {
            "position_error_norm_m": "Position error (m)",
            "velocity_error_norm_m_s": "Velocity error (m/s)",
            "body_rate_norm_rad_s": "Body rate (rad/s)",
            "applied_acceleration_norm_m_s2": "Applied acceleration (m/s²)",
            "applied_torque_norm_n_m": "Applied torque (N·m)",
        }
        times = [float(row["time_s"]) for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, constrained_layout=True)
        for label in response_labels:
            axes[0].plot(times, [float(row[label]) for row in rows], label=display_labels[label])
        axes[0].set_ylabel("Response (engineering units)")
        axes[0].grid(True, alpha=0.3)
        if response_labels:
            axes[0].legend(loc="best")
        for label in effort_labels:
            axes[1].plot(times, [float(row[label]) for row in rows], label=display_labels[label])
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Applied effort (engineering units)")
        axes[1].grid(True, alpha=0.3)
        if effort_labels:
            axes[1].legend(loc="best")
        case_label = str(run.get("case_name", "")).replace("_", " ")
        variant_label = str(run.get("variant_name", "")).replace("__", " · ").replace("_", " ")
        run_label = f"{case_label} / {variant_label}"
        fig.suptitle("Closed-loop response\n" + textwrap.fill(run_label, width=72))
        png_path = csv_path.with_suffix(".png")
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        run.setdefault("artifacts", {})["engineering_closed_loop_response_png"] = str(png_path)


def _run_distance(failed: dict[str, Any], passed: dict[str, Any]) -> float:
    distance = 0.0
    if failed.get("variant_name") != passed.get("variant_name"):
        distance += 2.0
    if failed.get("case_name") != passed.get("case_name"):
        distance += 1.0
    shared = set(dict(failed.get("metrics", {}) or {})) & set(dict(passed.get("metrics", {}) or {}))
    for name in shared:
        try:
            left = float(dict(failed.get("metrics", {}) or {})[name])
            right = float(dict(passed.get("metrics", {}) or {})[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(left) and math.isfinite(right):
            distance += abs(left - right) / max(abs(left), abs(right), 1.0)
    return float(distance)


def _first_divergence(run: dict[str, Any]) -> dict[str, Any] | None:
    engineering = dict(run.get("engineering", {}) or {})
    authority = dict(engineering.get("authority_analysis", {}) or {})
    if float(authority.get("saturation_fraction", 0.0) or 0.0) > 0.0:
        return {
            "kind": "actuator_saturation",
            "saturation_fraction": authority.get("saturation_fraction"),
            "first_saturation": authority.get("first_saturation"),
        }
    conformance = dict(engineering.get("stack_conformance", {}) or {})
    for check in list(conformance.get("checks", []) or []):
        if not bool(dict(check or {}).get("passed", True)):
            return {"kind": "stack_conformance", "check": check}
    failures = list(dict(run.get("failure_analysis", {}) or {}).get("failures", []) or [])
    return {"kind": "outcome_criterion", "failure": failures[0]} if failures else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ENGINEERING_SCHEMA",
    "EVIDENCE_MANIFEST_SCHEMA",
    "analyze_authority",
    "analyze_cadences",
    "analyze_completed_controller_bench",
    "analyze_frame_sign_probe",
    "analyze_stack_conformance",
    "analyze_stability_response",
    "build_aligned_flight_data_timeline",
    "lint_materialized_scenario",
    "render_closed_loop_response_plots",
    "reduce_controller_bench_failure",
    "write_evidence_manifest",
    "write_failure_reduction",
    "write_minimal_reproducer",
    "write_run_engineering_artifacts",
    "write_suite_engineering_exports",
]
