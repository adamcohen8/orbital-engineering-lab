"""Evidence-only scenario adapters for coverage and directed link analysis."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkEndpointHistory,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    evaluate_directed_link_sample,
    fixed_wgs84_site_history,
    spacecraft_endpoint_history,
    write_directed_link_artifacts,
)
from sim.analysis.global_coverage import (
    GlobalCoverageConfig,
    write_global_coverage_artifacts,
)
from sim.analysis.history_adapters import (
    AnalysisHistory,
    evaluate_history_global_coverage,
    history_from_single_run,
)
from sim.dynamics.orbit.frames import frame_context_from_mapping
from sim.numeric_backend import normalize_numeric_backend


def _only(mapping: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) in {path}: {', '.join(unknown)}")


def _artifact_paths(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: str(item) if isinstance(item, Path) else item
            for key, item in asdict(value).items()
            if key != "output_dir"
        }
    return {}


def _replace_derived_directory(path: Path, *, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved.exists():
        shutil.rmtree(resolved)


def _coverage_interval_rows(product: Any) -> list[dict[str, Any]]:
    if product.refined_intervals:
        return [asdict(row) for row in product.refined_intervals]
    sparse = product.cell_metrics.intervals
    rows: list[dict[str, Any]] = []
    for cell_offset, cell_index in enumerate(sparse.cell_index):
        first = int(sparse.interval_offset[cell_offset])
        stop = int(sparse.interval_offset[cell_offset + 1])
        for interval_index, sparse_index in enumerate(range(first, stop)):
            start_index = int(sparse.start_sample_index[sparse_index])
            end_index = int(sparse.end_sample_index_exclusive[sparse_index])
            start_censored = start_index == 0
            end_censored = end_index == product.times_s.size
            start_s = float(product.times_s[start_index])
            end_s = float(product.times_s[-1] if end_censored else product.times_s[end_index])
            rows.append({
                "cell_index": int(cell_index), "interval_index": interval_index,
                "start_s": start_s, "end_s": end_s, "duration_s": max(0.0, end_s - start_s),
                "start_censored": start_censored, "end_censored": end_censored,
                "acquisition_disposition": "study_start_censored" if start_censored else "sample_bounded",
                "loss_disposition": "study_end_censored" if end_censored else "sample_bounded",
                "acquisition_reason": "available" if start_censored else "not_covered",
                "loss_reason": "available" if end_censored else "not_covered",
            })
    return rows


def _terminal(value: Any, *, asset_id: str, parent_frame: str, path: str) -> LinkTerminal:
    data = dict(value or {})
    _only(data, {"terminal_id", "quat_parent_from_terminal", "quat_body_from_terminal", "pattern"}, path)
    pattern_data = dict(data.get("pattern", {}) or {})
    _only(pattern_data, {"kind", "gain_dbi", "half_angle_deg"}, f"{path}.pattern")
    kind = str(pattern_data.get("kind", "constant"))
    half_angle = pattern_data.get("half_angle_deg")
    return LinkTerminal(
        terminal_id=str(data.get("terminal_id") or ""),
        asset_id=asset_id,
        parent_frame=parent_frame,
        quat_parent_from_terminal=tuple(
            data.get(
                "quat_parent_from_terminal",
                data.get("quat_body_from_terminal", (1.0, 0.0, 0.0, 0.0)),
            )
        ),
        pattern=TerminalPattern(
            kind=kind,
            gain_dbi=float(pattern_data.get("gain_dbi")),
            half_angle_rad=None if half_angle is None else float(np.deg2rad(float(half_angle))),
        ),
    )


def _spacecraft_endpoint_at_time(
    history: AnalysisHistory,
    *,
    time_s: float,
    require_attitude: bool,
) -> LinkEndpointHistory:
    state = history.state_at(time_s)
    attitude = None if state.attitude_quat_bn is None else np.asarray([state.attitude_quat_bn])
    if require_attitude and attitude is None:
        raise ValueError(f"Directional terminal on {history.object_id!r} requires attitude evidence.")
    return spacecraft_endpoint_history(
        asset_id=history.object_id,
        state_provider_id=history.state_provider_id,
        times_s=np.asarray([time_s], dtype=float),
        positions_eci_km=np.asarray([state.position_eci_km], dtype=float),
        velocities_eci_km_s=np.asarray([state.velocity_eci_km_s], dtype=float),
        attitudes_quat_bn=attitude,
        attitude_source_kind=history.attitude_source_kind if attitude is not None else "not_required",
        attitude_provider_id=history.attitude_provider_id if attitude is not None else None,
    )


def _scenario_endpoint_history(
    *,
    kind: str,
    source: Any,
    terminal: LinkTerminal,
    times_s: np.ndarray,
    frame_context: Any,
    scenario_name: str,
) -> LinkEndpointHistory:
    if kind == "spacecraft":
        if np.array_equal(np.asarray(times_s, dtype=float), source.times_s):
            return source.link_endpoint(require_attitude=not terminal.pattern.attitude_independent)
        if np.asarray(times_s).size != 1:
            raise ValueError("Refined spacecraft endpoint requests must contain exactly one epoch.")
        return _spacecraft_endpoint_at_time(
            source,
            time_s=float(np.asarray(times_s, dtype=float)[0]),
            require_attitude=not terminal.pattern.attitude_independent,
        )
    return fixed_wgs84_site_history(
        asset_id=str(source.id),
        state_provider_id=f"scenario:{scenario_name}:ground_station:{source.id}",
        times_s=np.asarray(times_s, dtype=float),
        geodetic_latitude_deg=float(source.lat_deg),
        longitude_deg=float(source.lon_deg),
        ellipsoidal_height_km=float(source.alt_km),
        frame_context=frame_context,
    )


def _link_refinement_evaluator(
    *,
    config: DirectedLinkConfig,
    tx_spec: tuple[str, Any, LinkTerminal],
    rx_spec: tuple[str, Any, LinkTerminal],
    frame_context: Any,
    scenario_name: str,
) -> Any:
    def evaluate(time_s: float) -> tuple[bool, str]:
        sample_time = np.asarray([time_s], dtype=float)
        tx_kind, tx_source, tx_terminal = tx_spec
        rx_kind, rx_source, rx_terminal = rx_spec
        sample = evaluate_directed_link_sample(
            config,
            tx_history=_scenario_endpoint_history(
                kind=tx_kind,
                source=tx_source,
                terminal=tx_terminal,
                times_s=sample_time,
                frame_context=frame_context,
                scenario_name=scenario_name,
            ),
            rx_history=_scenario_endpoint_history(
                kind=rx_kind,
                source=rx_source,
                terminal=rx_terminal,
                times_s=sample_time,
                frame_context=frame_context,
                scenario_name=scenario_name,
            ),
            frame_context=frame_context,
        )
        return bool(sample.samples.available[0]), str(sample.samples.primary_reason[0])

    return evaluate


def _endpoint_refinement_source(*, kind: str, source: Any, scenario_name: str) -> str:
    if kind == "spacecraft":
        return f"{source.state_provider_id}:{source.refinement_source}"
    return f"scenario:{scenario_name}:ground_station:{source.id}:fixed_wgs84_analytic"


def _histories(context: Any, *, object_ids: set[str]) -> dict[str, AnalysisHistory]:
    attitude_enabled = bool(
        dict(dict(context.cfg.simulator.dynamics or {}).get("attitude", {}) or {}).get("enabled", True)
    )
    propagation = dict(getattr(context, "object_propagation", {}) or {})
    histories: dict[str, AnalysisHistory] = {}
    for object_id in sorted(object_ids):
        if object_id not in dict(context.truth_hist or {}):
            raise ValueError(f"Orbital analysis references object {object_id!r} without retained truth history.")
        truth = dict(context.truth_hist or {})[object_id]
        frame = str(dict(context.object_state_frames or {}).get(object_id, "eci") or "eci").lower()
        if frame != "eci":
            raise ValueError(f"Orbital analysis requires canonical ECI history for {object_id!r}; received {frame!r}.")
        product_kind = "ogp_scenario_history" if object_id in propagation else "onp_completed_run"
        object_config = dict(getattr(context.cfg, "objects", {}) or {}).get(object_id)
        object_attitude_enabled = product_kind != "ogp_scenario_history" and attitude_enabled and str(
            getattr(object_config, "runtime_profile", "") or ""
        ).strip().lower() != "trajectory_only"
        histories[object_id] = history_from_single_run(
            object_id=object_id,
            times_s=context.t_s,
            truth_state=truth,
            initial_jd_utc=float(context.cfg.simulator.initial_jd_utc),
            attitude_enabled=object_attitude_enabled,
            state_provider_id=f"scenario:{context.cfg.scenario_name}:{object_id}",
            product_kind=product_kind,
        )
    return histories


def run_scenario_orbital_analysis(*, context: Any) -> dict[str, Any]:
    section = context.cfg.outputs.orbital_analysis
    if not bool(section.enabled):
        return {}
    if context.cfg.simulator.initial_jd_utc is None:
        raise ValueError("outputs.orbital_analysis requires simulator.initial_jd_utc.")
    referenced_object_ids = {
        str(item.get("source_object_id") or "") for item in section.coverage
    }
    for item in section.directed_links:
        referenced_object_ids.update(
            str(item.get(key) or "")
            for key in ("tx_object_id", "rx_object_id")
            if str(item.get(key) or "").strip()
        )
    histories = _histories(context, object_ids=referenced_object_ids)
    frame_context = frame_context_from_mapping(
        dict(getattr(context.cfg.simulator, "frames", {}) or {}),
        jd_utc_start=context.cfg.simulator.initial_jd_utc,
        source="scenario_orbital_analysis",
    )
    root = Path(context.outdir) / "orbital_analysis"
    result: dict[str, Any] = {"schema_version": "oel.scenario-orbital-analysis.v1", "coverage": [], "directed_links": []}

    coverage_allowed = {
        "analysis_id", "source_object_id", "sensor_id", "order", "half_angle_deg",
        "quat_body_from_sensor", "max_range_km", "chunk_size", "max_working_memory_bytes",
        "max_cell_time_comparisons", "transition_time_tolerance_s", "transition_max_iterations",
        "max_transition_refinement_evaluations",
        "include_cell_csv", "include_fraction_plot",
        "numeric_backend",
    }
    for index, raw in enumerate(section.coverage):
        data = dict(raw)
        _only(data, coverage_allowed, f"outputs.orbital_analysis.coverage[{index}]")
        source_id = str(data.get("source_object_id") or "")
        if source_id not in histories:
            raise ValueError(f"Coverage source_object_id {source_id!r} is not an active scenario object.")
        history = histories[source_id]
        if history.attitude_quat_bn is None:
            raise ValueError("Scenario coverage requires achieved attitude; simulator attitude dynamics are disabled.")
        config = GlobalCoverageConfig(
            numeric_backend=normalize_numeric_backend(data.get("numeric_backend", "rust")),
            analysis_id=str(data.get("analysis_id") or ""), source_asset_id=source_id,
            state_provider_id=history.state_provider_id, attitude_source_kind="achieved",
            attitude_provider_id=str(history.attitude_provider_id), sensor_id=str(data.get("sensor_id") or ""),
            order=int(data["order"]), half_angle_rad=float(np.deg2rad(float(data.get("half_angle_deg")))),
            quat_body_from_sensor=tuple(data.get("quat_body_from_sensor", (1.0, 0.0, 0.0, 0.0))),
            max_range_km=data.get("max_range_km"), chunk_size=int(data.get("chunk_size", 8192)),
            max_working_memory_bytes=int(data.get("max_working_memory_bytes", 512 * 1024 * 1024)),
            max_cell_time_comparisons=int(data.get("max_cell_time_comparisons", 300_000_000)),
            max_transition_refinement_evaluations=int(
                data.get("max_transition_refinement_evaluations", 5_000_000)
            ),
            transition_time_tolerance_s=(
                None
                if "transition_time_tolerance_s" not in data
                else float(data["transition_time_tolerance_s"])
            ),
            transition_max_iterations=(
                None
                if "transition_max_iterations" not in data
                else int(data["transition_max_iterations"])
            ),
        )
        product = evaluate_history_global_coverage(
            config, history=history, frame_context=frame_context
        )
        destination = root / "coverage" / config.analysis_id
        _replace_derived_directory(destination, root=Path(context.outdir))
        artifacts = write_global_coverage_artifacts(
            product,
            destination,
            include_cell_csv=bool(data.get("include_cell_csv", False)),
            include_fraction_plot=bool(data.get("include_fraction_plot", True)),
            plot_scenario_name=str(context.cfg.scenario_name or ""),
        )
        result["coverage"].append({
            "analysis_id": config.analysis_id, "source_object_id": source_id,
            "state_provider_id": history.state_provider_id, "product_kind": history.product_kind,
            "attitude_source_kind": history.attitude_source_kind,
            "attitude_provider_id": history.attitude_provider_id,
            "refinement_source": product.refinement_provider_id or "sample_bounded",
            "summary": product.summary,
            "samples": [
                {"sample_index": i, "time_s": float(time_s), "covered_cell_count": int(product.covered_cell_count[i]),
                 "instantaneous_covered_fraction": float(product.instantaneous_covered_fraction[i])}
                for i, time_s in enumerate(product.times_s)
            ],
            "intervals": _coverage_interval_rows(product),
            "transitions": [asdict(row) for row in product.refined_transitions],
            "artifacts": _artifact_paths(artifacts),
            "input_evidence_sha256": product.input_evidence_sha256,
            "semantic_sha256": product.interval_semantic_sha256,
        })

    link_allowed = {
        "analysis_id", "link_id", "tx_object_id", "rx_object_id", "tx_ground_station_id",
        "rx_ground_station_id", "tx_terminal", "rx_terminal",
        "carrier_frequency_hz", "tx_power_w", "data_rate_bps", "system_noise_temperature_k",
        "required_eb_n0_db", "tx_line_loss_db", "rx_line_loss_db", "misc_loss_db", "max_range_km",
        "min_fixed_site_elevation_deg", "transition_time_tolerance_s", "transition_max_iterations",
        "include_margin_plot",
        "numeric_backend",
    }
    stations = {
        str(station.id): station
        for station in list(context.cfg.ground_stations or [])
        if bool(station.enabled)
    }
    for index, raw in enumerate(section.directed_links):
        data = dict(raw)
        _only(data, link_allowed, f"outputs.orbital_analysis.directed_links[{index}]")
        endpoint_specs: dict[str, tuple[str, str, Any]] = {}
        for endpoint in ("tx", "rx"):
            object_id = str(data.get(f"{endpoint}_object_id") or "").strip()
            station_id = str(data.get(f"{endpoint}_ground_station_id") or "").strip()
            if object_id:
                endpoint_specs[endpoint] = ("spacecraft", object_id, histories[object_id])
            else:
                endpoint_specs[endpoint] = ("fixed_wgs84_site", station_id, stations[station_id])
        tx_kind, tx_id, tx_source = endpoint_specs["tx"]
        rx_kind, rx_id, rx_source = endpoint_specs["rx"]
        tx_terminal = _terminal(
            data.get("tx_terminal"),
            asset_id=tx_id,
            parent_frame="body" if tx_kind == "spacecraft" else "enu",
            path=f"directed_links[{index}].tx_terminal",
        )
        rx_terminal = _terminal(
            data.get("rx_terminal"),
            asset_id=rx_id,
            parent_frame="body" if rx_kind == "spacecraft" else "enu",
            path=f"directed_links[{index}].rx_terminal",
        )
        fixed_station = tx_source if tx_kind == "fixed_wgs84_site" else rx_source if rx_kind == "fixed_wgs84_site" else None
        minimum_elevation_deg = data.get("min_fixed_site_elevation_deg")
        if minimum_elevation_deg is None and fixed_station is not None:
            minimum_elevation_deg = fixed_station.min_elevation_deg
        maximum_range_km = data.get("max_range_km")
        if maximum_range_km is None and fixed_station is not None:
            maximum_range_km = fixed_station.max_range_km
        config = DirectedLinkConfig(
            numeric_backend=normalize_numeric_backend(data.get("numeric_backend", "rust")),
            analysis_id=str(data.get("analysis_id") or ""), link_id=str(data.get("link_id") or ""),
            tx_terminal=tx_terminal, rx_terminal=rx_terminal,
            carrier_frequency_hz=float(data.get("carrier_frequency_hz")), tx_power_w=float(data.get("tx_power_w")),
            data_rate_bps=float(data.get("data_rate_bps")),
            system_noise_temperature_k=float(data.get("system_noise_temperature_k")),
            required_eb_n0_db=float(data.get("required_eb_n0_db")),
            tx_line_loss_db=float(data.get("tx_line_loss_db", 0.0)),
            rx_line_loss_db=float(data.get("rx_line_loss_db", 0.0)), misc_loss_db=float(data.get("misc_loss_db", 0.0)),
            min_fixed_site_elevation_rad=(
                None if minimum_elevation_deg is None else float(np.deg2rad(float(minimum_elevation_deg)))
            ),
            max_range_km=maximum_range_km,
            transition_time_tolerance_s=(
                None
                if "transition_time_tolerance_s" not in data
                else float(data["transition_time_tolerance_s"])
            ),
            transition_max_iterations=(
                None
                if "transition_max_iterations" not in data
                else int(data["transition_max_iterations"])
            ),
        )
        scenario_name = str(context.cfg.scenario_name or "")
        sample_times = np.asarray(context.t_s, dtype=float)
        tx_history = _scenario_endpoint_history(
            kind=tx_kind,
            source=tx_source,
            terminal=tx_terminal,
            times_s=sample_times,
            frame_context=frame_context,
            scenario_name=scenario_name,
        )
        rx_history = _scenario_endpoint_history(
            kind=rx_kind,
            source=rx_source,
            terminal=rx_terminal,
            times_s=sample_times,
            frame_context=frame_context,
            scenario_name=scenario_name,
        )

        refine = config.transition_time_tolerance_s is not None
        evaluator_at_time = _link_refinement_evaluator(
            config=config,
            tx_spec=(tx_kind, tx_source, tx_terminal),
            rx_spec=(rx_kind, rx_source, rx_terminal),
            frame_context=frame_context,
            scenario_name=scenario_name,
        )
        refinement_provider_id = (
            f"tx={_endpoint_refinement_source(kind=tx_kind, source=tx_source, scenario_name=scenario_name)};"
            f"rx={_endpoint_refinement_source(kind=rx_kind, source=rx_source, scenario_name=scenario_name)}"
        )
        product = evaluate_directed_link(
            config,
            tx_history=tx_history,
            rx_history=rx_history,
            frame_context=frame_context,
            evaluator_at_time=evaluator_at_time if refine else None,
            refinement_provider_id=refinement_provider_id if refine else None,
        )
        destination = root / "directed_links" / config.analysis_id
        _replace_derived_directory(destination, root=Path(context.outdir))
        artifacts = write_directed_link_artifacts(
            product,
            destination,
            include_margin_plot=bool(data.get("include_margin_plot", True)),
            plot_scenario_name=str(context.cfg.scenario_name or ""),
        )
        result["directed_links"].append({
            "analysis_id": config.analysis_id, "link_id": config.link_id,
            "tx_object_id": tx_id, "rx_object_id": rx_id,
            "tx_endpoint_kind": tx_kind, "rx_endpoint_kind": rx_kind,
            "tx_terminal_parent_frame": tx_terminal.parent_frame,
            "rx_terminal_parent_frame": rx_terminal.parent_frame,
            "tx_state_provider_id": tx_history.state_provider_id, "rx_state_provider_id": rx_history.state_provider_id,
            "tx_attitude_source_kind": tx_history.attitude_source_kind,
            "tx_attitude_provider_id": tx_history.attitude_provider_id,
            "rx_attitude_source_kind": rx_history.attitude_source_kind,
            "rx_attitude_provider_id": rx_history.attitude_provider_id,
            "refinement_source": {
                "method": product.summary["transition_refinement"]["method"],
                "provider_id": product.refinement_provider_id,
            },
            "summary": product.summary,
            "samples": [
                {"sample_index": i, "time_s": float(product.samples.time_s[i]), "range_km": float(product.samples.range_km[i]),
                 "margin_db": float(product.samples.margin_db[i]), "available": bool(product.samples.available[i]),
                 "primary_reason": product.samples.primary_reason[i]}
                for i in range(product.samples.time_s.size)
            ],
            "windows": [asdict(row) for row in product.windows],
            "transitions": [asdict(row) for row in product.transitions],
            "artifacts": _artifact_paths(artifacts), "input_evidence_sha256": product.input_evidence_sha256,
            "semantic_sha256": product.semantic_sha256,
        })
    if section.access_chains:
        from sim.analysis.access_chain_artifacts import write_access_chain_artifacts
        from sim.analysis.access_chain_sensor import evaluate_sensor_gate
        from sim.analysis.access_chains import Link, evaluate_chain, parse_chain

        catalog = {row["analysis_id"]: Link.from_scenario(row) for row in section.directed_links}
        coverage_config = {row["analysis_id"]: row for row in section.coverage}
        coverage_catalog = {key: (row["source_object_id"], row["order"]) for key, row in coverage_config.items()}
        link_rows = {row["analysis_id"]: row for row in result["directed_links"]}
        coverage_rows = {row["analysis_id"]: row for row in result["coverage"]}
        result["access_chains"] = []
        for raw in section.access_chains:
            chain = parse_chain(raw, catalog, coverage_catalog)
            gate = chain.sensor_gate
            sensor = None
            if gate is not None:
                config = coverage_config[gate.coverage_analysis_id]
                source = coverage_rows[gate.coverage_analysis_id]
                cell_windows = None if gate.cell_index is None else [
                    row for row in source["intervals"] if row["cell_index"] == gate.cell_index
                ]
                sensor = evaluate_sensor_gate(
                    gate, config, histories[config["source_object_id"]], frame_context,
                    cell_windows=cell_windows, source_digest=source["semantic_sha256"],
                )
            evidence = evaluate_chain(chain, catalog, link_rows, sensor=sensor)
            destination = root / "access_chains" / chain.analysis_id
            _replace_derived_directory(destination, root=Path(context.outdir))
            artifacts = write_access_chain_artifacts(
                evidence, destination, include_timeline=chain.include_timeline,
                scenario_name=str(context.cfg.scenario_name or ""),
            )
            result["access_chains"].append({**evidence, "artifacts": artifacts})
    return result


__all__ = ["run_scenario_orbital_analysis"]
