"""Evaluate an access-chain request against a completed run's review evidence."""

import argparse
import hashlib
import json
from bisect import bisect_left
from pathlib import Path

import numpy as np

from sim.analysis.access_chain_artifacts import write_access_chain_artifacts
from sim.analysis.access_chain_sensor import evaluate_sensor_gate
from sim.analysis.access_chains import Link, evaluate_chain, parse_chain
from sim.analysis.history_adapters import history_from_single_run
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.review.workspace import ReviewWorkspace


def _rows(workspace, sql, parameters=(), *, max_rows=250_000):
    result = workspace.query(sql, parameters, max_rows=max_rows, max_vm_steps=12_000_000)
    if result.truncated:
        raise ValueError("Access-chain review evidence exceeds the query budget.")
    return result.rows


def _link_evidence(workspace, analysis_id, metadata, sample_count):
    params = (analysis_id,)
    samples = _rows(workspace, """
        SELECT sample_index, time_s, margin_db, available, primary_reason FROM link_samples
        WHERE analysis_id = ? ORDER BY sample_index
    """, params)
    if len(samples) != sample_count or any(row["sample_index"] != i for i, row in enumerate(samples)):
        raise ValueError(f"Link {analysis_id!r} has incomplete retained sample evidence.")
    windows = _rows(workspace, """
        SELECT interval_index, start_s, end_s, start_censored, end_censored,
               acquisition_disposition, loss_disposition
        FROM link_windows WHERE analysis_id = ? ORDER BY interval_index
    """, params)
    summary = json.loads(metadata["summary_json"])
    if summary["status"] != "complete" or len(windows) != summary["interval_count"]:
        raise ValueError(f"Link {analysis_id!r} has incomplete window evidence.")
    transitions = _rows(workspace, """
        SELECT time_s, transition_kind, reason_before, reason_after
        FROM link_transitions WHERE analysis_id = ? ORDER BY time_s
    """, params)
    by_boundary = {(row["transition_kind"], row["time_s"]): row for row in transitions}
    times = [row["time_s"] for row in samples]
    for window in windows:
        start, end = window["start_s"], window["end_s"]
        acquisition = by_boundary.get(("acquisition", start))
        loss = by_boundary.get(("loss", end))
        window["acquisition_reason"] = (
            "available" if window["start_censored"] else (
                acquisition["reason_before"] if acquisition else
                samples[max(0, bisect_left(times, start) - 1)]["primary_reason"]
            )
        )
        window["loss_reason"] = (
            "available" if window["end_censored"] else (
                loss["reason_after"] if loss else samples[min(len(times) - 1, bisect_left(times, end))]["primary_reason"]
            )
        )
    return {**metadata, "samples": samples, "windows": windows}


def _sensor_evidence(workspace, gate, config, simulator, coverage, samples):
    source = coverage["source_object_id"]
    frames = _rows(workspace, "SELECT state_frame FROM object_state_frame WHERE object_id = ?", (source,))
    if coverage["attitude_source_kind"] != "achieved" or frames != [{"state_frame": "eci"}]:
        raise ValueError("Sensor gate requires canonical ECI state and achieved attitude evidence.")
    rows = _rows(workspace, """
        SELECT sample_index, time_s, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km,
               vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s, quat_w, quat_x, quat_y, quat_z
        FROM object_state WHERE object_id = ? ORDER BY sample_index
    """, (source,))
    if len(rows) != samples or any(row["sample_index"] != i for i, row in enumerate(rows)):
        raise ValueError(f"Incomplete retained state for coverage source {source!r}.")
    history = history_from_single_run(
        object_id=source, times_s=np.asarray([row["time_s"] for row in rows]),
        truth_state=np.asarray([list(row.values())[2:] for row in rows], dtype=float),
        initial_jd_utc=simulator["initial_jd_utc"], attitude_enabled=True,
        state_provider_id=coverage["state_provider_id"], product_kind=coverage["product_kind"],
    )
    frame = frame_context_from_mapping(
        simulator.get("frames") or {}, jd_utc_start=history.initial_jd_utc, source="scenario_orbital_analysis",
    )
    cell_windows = None
    if gate.cell_index is not None:
        cell_windows = _rows(workspace, """
            SELECT interval_index, start_s, end_s, start_censored, end_censored,
                   acquisition_disposition, loss_disposition, acquisition_reason, loss_reason
            FROM coverage_intervals WHERE analysis_id = ? AND cell_index = ? ORDER BY interval_index
        """, (gate.coverage_analysis_id, gate.cell_index))
    return evaluate_sensor_gate(
        gate, config, history, frame, cell_windows=cell_windows, source_digest=coverage["semantic_sha256"],
    )


def evaluate_completed_run(path, request, *, exclude_ground_stations=()):
    """Read persisted review evidence and return an independently replayable chain result."""
    with ReviewWorkspace.open(path) as workspace:
        metadata, = _rows(workspace, """
            SELECT run_id, scenario_name, samples, config_json, config_sha256 FROM run_metadata
        """, max_rows=1)
        config_json = metadata["config_json"]
        if not config_json or hashlib.sha256(config_json.encode("utf-8")).hexdigest() != metadata["config_sha256"]:
            raise ValueError("The review store's normalized configuration digest does not match.")
        config = json.loads(config_json)
        section = config.get("outputs", {}).get("orbital_analysis", {})
        link_records = {row["analysis_id"]: row for row in _rows(workspace, "SELECT * FROM link_summary")}
        if not link_records:
            raise ValueError("The completed run has no retained directed-link evidence.")
        catalog = {
            row["analysis_id"]: Link.from_scenario(row)
            for row in section.get("directed_links", []) if row["analysis_id"] in link_records
        }
        cover_records = {row["analysis_id"]: row for row in _rows(workspace, "SELECT * FROM coverage_summary")}
        cover_config = {
            row["analysis_id"]: row for row in section.get("coverage", []) if row["analysis_id"] in cover_records
        }
        chain = parse_chain(request, catalog, {
            key: (row["source_object_id"], row["order"]) for key, row in cover_config.items()
        })
        link_ids = sorted({link_id for route in chain.paths for link_id in route.links})
        links = {}
        for link_id in link_ids:
            record = link_records[link_id]
            source = record["tx_endpoint_kind"], record["tx_object_id"]
            destination = record["rx_endpoint_kind"], record["rx_object_id"]
            if (source, destination) != (catalog[link_id].source, catalog[link_id].destination):
                raise ValueError(f"Link {link_id!r} endpoints disagree with the run configuration.")
            links[link_id] = _link_evidence(workspace, link_id, record, metadata["samples"])
        sensor = None
        if chain.sensor_gate is not None:
            coverage_id = chain.sensor_gate.coverage_analysis_id
            sensor = _sensor_evidence(
                workspace, chain.sensor_gate, cover_config[coverage_id], config["simulator"],
                cover_records[coverage_id], metadata["samples"],
            )
        result = evaluate_chain(
            chain, catalog, links, sensor=sensor, exclude_ground_stations=exclude_ground_stations,
        )
        result["source_run"] = {
            key: metadata[key] for key in ("run_id", "scenario_name", "config_sha256")
        }
        return chain, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed output directory or review database.")
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--request", type=Path, help="JSON file containing one access-chain definition.")
    request.add_argument("--request-json", help="One inline JSON access-chain definition.")
    parser.add_argument("--exclude-ground-station", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the derived artifacts.")
    args = parser.parse_args(argv)
    try:
        raw = args.request.read_text(encoding="utf-8") if args.request else args.request_json
        chain, result = evaluate_completed_run(
            args.run, json.loads(raw), exclude_ground_stations=args.exclude_ground_station,
        )
        # Publish to a new directory only. Existing evidence is never overwritten.
        artifacts = write_access_chain_artifacts(
            result, args.output_dir, include_timeline=chain.include_timeline,
            scenario_name=result["source_run"]["scenario_name"],
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"summary": result["summary"], "artifacts": artifacts}, sort_keys=True))
    return 0
