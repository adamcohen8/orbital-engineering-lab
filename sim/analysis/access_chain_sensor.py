"""Evaluate a named scenario sensor at one WGS84 point or coverage cell center."""

from dataclasses import asdict

import numpy as np

from sim.analysis.event_refinement import availability_intervals, refine_availability_transitions
from sim.analysis.global_coverage import _prepare_ecef_state
from sim.analysis.healpix import cached_healpix_wgs84_centers
from sim.analysis.observer_target_geometry import evaluate_surface_targets_ecef
from sim.utils.geodesy import WGS84_A_KM, WGS84_B_KM, geodetic_to_ecef_km


def evaluate_sensor_gate(gate, config, history, frame_context, *, cell_windows=None, source_digest=None):
    if history.attitude_quat_bn is None or history.attitude_source_kind != "achieved":
        raise ValueError("Access-chain sensor gates require retained achieved attitude.")
    if gate.cell_index is None:
        target = geodetic_to_ecef_km(gate.latitude_deg, gate.longitude_deg, 0.0)
        normal = target / np.square([WGS84_A_KM, WGS84_A_KM, WGS84_B_KM])
        targets = target[None, :]
        normals = (normal / np.linalg.norm(normal))[None, :]
    else:
        cell = cached_healpix_wgs84_centers(config["order"], np.asarray([gate.cell_index], dtype=np.int64))
        targets, normals = cell.ecef_km, cell.outward_normal_ecef
    mounting = np.asarray(config.get("quat_body_from_sensor", [1, 0, 0, 0]), dtype=float)
    half_angle = np.deg2rad(float(config["half_angle_deg"]))
    max_range = config.get("max_range_km")
    backend = config.get("numeric_backend", "rust") if gate.cell_index is not None else "python"

    def geometry(observer, boresight):
        result = evaluate_surface_targets_ecef(
            observer_ecef_km=observer, boresight_ecef=boresight,
            target_ecef_km=targets, target_outward_normal_ecef=normals,
            half_angle_rad=half_angle, max_range_km=max_range, numeric_backend=backend,
        )
        return bool(result.available[0])

    times = history.times_s
    positions, boresights = _prepare_ecef_state(
        times, history.position_eci_km, history.attitude_quat_bn, mounting, frame_context,
    )
    available = [geometry(pos, axis) for pos, axis in zip(positions, boresights, strict=True)]

    def evaluate(time):
        state = history.state_at(time)
        pos, axis = _prepare_ecef_state(
            np.asarray([time]), state.position_eci_km[None, :],
            state.attitude_quat_bn[None, :], mounting, frame_context,
        )
        passes = geometry(pos[0], axis[0])
        return passes, "available" if passes else "not_covered"

    windows = cell_windows
    if windows is None:
        reasons = ["available" if passes else "not_covered" for passes in available]
        transitions = ()
        if "transition_time_tolerance_s" in config:
            transitions = refine_availability_transitions(
                times, np.asarray(available), reasons, evaluator_at_time=evaluate,
                time_tolerance_s=float(config["transition_time_tolerance_s"]),
                max_iterations=config["transition_max_iterations"], numeric_backend="python",
            )
        windows = [asdict(row) for row in availability_intervals(
            times, np.asarray(available), reasons, transitions=transitions,
        )]
    return {
        "times_s": times.tolist(),
        "available": available,
        "windows": windows,
        "provenance": {
            **asdict(gate), "sensor_id": config["sensor_id"], "source_object_id": history.object_id,
            "target_kind": "wgs84_surface_point" if gate.cell_index is None else "healpix_cell_center",
            "target_ecef_km": targets[0].tolist(),
            "coverage_semantic_sha256": source_digest,
            "state_provider_id": history.state_provider_id,
            "attitude_provider_id": history.attitude_provider_id,
            "refinement_source": history.refinement_source,
        },
    }
