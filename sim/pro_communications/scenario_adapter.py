"""Stable adapters from completed OEL scenario evidence to communications products."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    fixed_wgs84_site_history,
    spacecraft_endpoint_history,
)
from sim.api import SimulationConfig
from sim.dynamics.orbit.frames import FrameContext
from sim.pro_communications.campaign import LinkGeometryEvidence
from sim.pro_communications.contracts import AntennaPatternProfile, LinkBudgetProfile, ResourceLimits
from sim.pro_communications.coverage import (
    ProCommunicationsCoverageConfig,
    ProCommunicationsCoverageResult,
    evaluate_pro_communications_coverage,
)
from sim.pro_communications.link_budget import ProLinkBudgetResult, evaluate_pro_link_budget
from sim.review import ReviewWorkspace

PRO_SCENARIO_ADAPTER_VERSION = "oel.pro-communications-scenario-adapter.v0.2"
_RUN_METADATA_QUERY = """
SELECT
    scenario_name,
    config_sha256,
    config_json
FROM run_metadata
""".strip()
_STATE_QUERY = """
SELECT
    time_s,
    pos_x_eci_km,
    pos_y_eci_km,
    pos_z_eci_km,
    vel_x_eci_km_s,
    vel_y_eci_km_s,
    vel_z_eci_km_s,
    quat_w,
    quat_x,
    quat_y,
    quat_z
FROM object_state
WHERE object_id = ?
ORDER BY time_s
""".strip()


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompletedScenarioObjectEvidence:
    scenario_name: str
    scenario_path: Path
    scenario_sha256: str
    review_path: Path
    review_sha256: str
    normalized_config_sha256: str
    object_id: str
    times_s: np.ndarray
    positions_eci_km: np.ndarray
    velocities_eci_km_s: np.ndarray
    attitudes_quat_bn: np.ndarray | None
    attitude_source_kind: str
    attitude_provider_id: str | None
    frame_context: FrameContext
    ground_stations: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ProScenarioFixedSiteLinkRequest:
    analysis_id: str
    object_id: str
    station_id: str
    profile: LinkBudgetProfile
    direction: str = "spacecraft_to_earth"
    attitude_source_kind: str = "not_required"
    attitude_provider_id: str | None = None
    quat_body_from_spacecraft_terminal: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for field_name in ("analysis_id", "object_id", "station_id"):
            object.__setattr__(self, field_name, _required(getattr(self, field_name), field_name))
        if not isinstance(self.profile, LinkBudgetProfile):
            raise ValueError("profile must be a LinkBudgetProfile.")
        direction = str(self.direction or "").strip().lower()
        if direction not in {"spacecraft_to_earth", "earth_to_spacecraft"}:
            raise ValueError("direction must be spacecraft_to_earth or earth_to_spacecraft.")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class ProScenarioLinkProduct:
    request: ProScenarioFixedSiteLinkRequest
    source: CompletedScenarioObjectEvidence
    geometry: LinkGeometryEvidence
    result: ProLinkBudgetResult
    provenance: Mapping[str, Any]


def _terminal_pattern(profile: AntennaPatternProfile) -> TerminalPattern:
    if profile.kind == "constant":
        return TerminalPattern(kind="constant", gain_dbi=profile.peak_gain_dbi)
    if profile.kind == "axisymmetric_hard_cone":
        half_angle = float(profile.half_angle_deg)
    else:
        half_angle = float(profile.off_axis_deg[-1])
    return TerminalPattern(
        kind="axisymmetric_hard_cone",
        gain_dbi=profile.peak_gain_dbi,
        half_angle_rad=np.deg2rad(half_angle),
    )


def load_completed_scenario_object(
    scenario_path: str | Path,
    *,
    object_id: str,
    attitude_source_kind: str,
    attitude_provider_id: str | None = None,
    workspace_root: str | Path | None = None,
    resource_limits: ResourceLimits | None = None,
) -> CompletedScenarioObjectEvidence:
    """Load one complete object history without executing or mutating the scenario."""

    root = Path(workspace_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    source_path = Path(scenario_path).expanduser().resolve()
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("scenario_path must resolve inside the trusted workspace.") from exc
    scenario = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("Scenario YAML must resolve to an object.")
    normalized_config_json = json.dumps(
        SimulationConfig.from_yaml(source_path).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    normalized_config_sha256 = hashlib.sha256(normalized_config_json.encode("utf-8")).hexdigest()
    normalized_object_id = _required(object_id, "object_id")
    if normalized_object_id not in scenario.get("objects", {}):
        raise ValueError(f"Object {normalized_object_id!r} is absent from the scenario.")
    kind = str(attitude_source_kind or "").strip().lower()
    if kind not in {"achieved", "replay", "not_required"}:
        raise ValueError("Scenario adapter attitude_source_kind must be achieved, replay, or not_required.")
    attitude_enabled = bool(scenario["simulator"]["dynamics"]["attitude"].get("enabled", False))
    if kind == "achieved" and not attitude_enabled:
        raise ValueError("Achieved attitude was requested but scenario attitude dynamics are disabled.")
    if kind == "not_required":
        if attitude_provider_id is not None:
            raise ValueError("attitude_provider_id must be absent when attitude is not required.")
        provider = None
    else:
        provider = _required(attitude_provider_id, "attitude_provider_id")
    scenario_output = (root / str(scenario["outputs"]["output_dir"])).resolve()
    try:
        scenario_output.relative_to(root)
    except ValueError as exc:
        raise ValueError("Scenario output_dir must resolve inside the trusted workspace.") from exc
    compact = scenario_output / "review" / "run.sqlite.gz"
    ordinary = scenario_output / "review" / "run.sqlite"
    review_path = compact if compact.is_file() else ordinary
    if not review_path.is_file():
        raise FileNotFoundError(
            f"Completed review evidence is missing for {source_path}; validate and run the scenario first."
        )
    limits = resource_limits or ResourceLimits()
    with ReviewWorkspace.open(scenario_output) as workspace:
        metadata = workspace.query(_RUN_METADATA_QUERY, max_rows=2)
        if metadata.truncated or metadata.row_count != 1:
            raise RuntimeError("Expected exactly one complete run_metadata row in the scenario review evidence.")
        metadata_row = metadata.rows[0]
        stored_config_json = str(metadata_row["config_json"] or "")
        stored_config_sha256 = str(metadata_row["config_sha256"] or "")
        if hashlib.sha256(stored_config_json.encode("utf-8")).hexdigest() != stored_config_sha256:
            raise RuntimeError("Scenario review metadata config_json does not match its recorded SHA-256.")
        if stored_config_sha256 != normalized_config_sha256:
            raise RuntimeError(
                "Scenario review evidence does not match the current normalized scenario; validate and rerun it."
            )
        if str(metadata_row["scenario_name"] or "") != _required(scenario.get("scenario_name"), "scenario_name"):
            raise RuntimeError("Scenario review evidence has a different scenario_name than the current scenario.")
        query = workspace.query(_STATE_QUERY, (normalized_object_id,), max_rows=limits.max_time_samples + 1)
    if query.truncated or query.row_count < 2:
        raise RuntimeError("Expected a complete multi-sample object history within the resource profile.")
    limits.enforce(time_samples=query.row_count)
    times = np.asarray([row["time_s"] for row in query.rows], dtype=float)
    positions = np.asarray(
        [[row["pos_x_eci_km"], row["pos_y_eci_km"], row["pos_z_eci_km"]] for row in query.rows],
        dtype=float,
    )
    velocities = np.asarray(
        [[row["vel_x_eci_km_s"], row["vel_y_eci_km_s"], row["vel_z_eci_km_s"]] for row in query.rows],
        dtype=float,
    )
    attitudes = (
        None
        if kind == "not_required"
        else np.asarray(
            [[row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]] for row in query.rows],
            dtype=float,
        )
    )
    stations = {_required(value.get("id"), "ground_station.id"): value for value in scenario.get("ground_stations", [])}
    return CompletedScenarioObjectEvidence(
        scenario_name=_required(scenario.get("scenario_name"), "scenario_name"),
        scenario_path=source_path,
        scenario_sha256=_sha256(source_path),
        review_path=review_path,
        review_sha256=_sha256(review_path),
        normalized_config_sha256=normalized_config_sha256,
        object_id=normalized_object_id,
        times_s=times,
        positions_eci_km=positions,
        velocities_eci_km_s=velocities,
        attitudes_quat_bn=attitudes,
        attitude_source_kind=kind,
        attitude_provider_id=provider,
        frame_context=FrameContext(jd_utc_start=float(scenario["simulator"]["initial_jd_utc"])),
        ground_stations=stations,
    )


def evaluate_scenario_fixed_site_link(
    scenario_path: str | Path,
    request: ProScenarioFixedSiteLinkRequest,
    *,
    workspace_root: str | Path | None = None,
    resource_limits: ResourceLimits | None = None,
) -> ProScenarioLinkProduct:
    """Evaluate one Pro fixed-site link from a completed deterministic scenario."""

    if not isinstance(request, ProScenarioFixedSiteLinkRequest):
        raise ValueError("request must be ProScenarioFixedSiteLinkRequest.")
    source = load_completed_scenario_object(
        scenario_path,
        object_id=request.object_id,
        attitude_source_kind=request.attitude_source_kind,
        attitude_provider_id=request.attitude_provider_id,
        workspace_root=workspace_root,
        resource_limits=resource_limits,
    )
    station = source.ground_stations.get(request.station_id)
    if station is None:
        raise ValueError(f"Ground station {request.station_id!r} is absent from the scenario.")
    spacecraft = spacecraft_endpoint_history(
        asset_id=request.object_id,
        state_provider_id=f"review:{source.scenario_name}:{request.object_id}",
        times_s=source.times_s,
        positions_eci_km=source.positions_eci_km,
        velocities_eci_km_s=source.velocities_eci_km_s,
        attitudes_quat_bn=source.attitudes_quat_bn,
        attitude_source_kind=request.attitude_source_kind,
        attitude_provider_id=request.attitude_provider_id,
    )
    site = fixed_wgs84_site_history(
        asset_id=request.station_id,
        state_provider_id=f"fixed_wgs84:{request.station_id}",
        times_s=source.times_s,
        geodetic_latitude_deg=float(station["lat_deg"]),
        longitude_deg=float(station["lon_deg"]),
        ellipsoidal_height_km=float(station["alt_km"]),
        frame_context=source.frame_context,
    )
    if request.direction == "spacecraft_to_earth":
        tx_history, rx_history = spacecraft, site
        tx_asset, rx_asset = request.object_id, request.station_id
        tx_parent, rx_parent = "body", "enu"
        tx_mount = request.quat_body_from_spacecraft_terminal
        rx_mount = (1.0, 0.0, 0.0, 0.0)
    else:
        tx_history, rx_history = site, spacecraft
        tx_asset, rx_asset = request.station_id, request.object_id
        tx_parent, rx_parent = "enu", "body"
        tx_mount = (1.0, 0.0, 0.0, 0.0)
        rx_mount = request.quat_body_from_spacecraft_terminal
    reference = evaluate_directed_link(
        DirectedLinkConfig(
            analysis_id=f"{request.analysis_id}:geometry",
            link_id=f"{tx_asset}-to-{rx_asset}",
            tx_terminal=LinkTerminal(
                terminal_id=request.profile.transmitter.transmitter_id,
                asset_id=tx_asset,
                parent_frame=tx_parent,
                quat_parent_from_terminal=tx_mount,
                pattern=_terminal_pattern(request.profile.transmitter.antenna),
            ),
            rx_terminal=LinkTerminal(
                terminal_id=request.profile.receiver.receiver_id,
                asset_id=rx_asset,
                parent_frame=rx_parent,
                quat_parent_from_terminal=rx_mount,
                pattern=_terminal_pattern(request.profile.receiver.antenna),
            ),
            carrier_frequency_hz=request.profile.carrier_frequency_hz,
            tx_power_w=request.profile.transmitter.power_w,
            data_rate_bps=request.profile.waveform.data_rate_bps,
            system_noise_temperature_k=request.profile.receiver.resolved_system_noise_temperature_k,
            required_eb_n0_db=request.profile.waveform.required_eb_n0_db,
            tx_line_loss_db=request.profile.transmitter.line_loss_db,
            rx_line_loss_db=request.profile.receiver.line_loss_db,
            misc_loss_db=(
                request.profile.legacy_misc_loss_db
                + request.profile.polarization_loss_db
                + request.profile.pointing_loss_db
                + request.profile.implementation_loss_db
            ),
            min_fixed_site_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
            max_range_km=float(station["max_range_km"]),
        ),
        tx_history=tx_history,
        rx_history=rx_history,
        frame_context=source.frame_context,
    )
    geometry = LinkGeometryEvidence(
        source_id=reference.input_evidence_sha256,
        times_s=reference.samples.time_s,
        range_km=reference.samples.range_km,
        elevation_rad=reference.samples.fixed_site_elevation_rad,
        tx_off_axis_rad=reference.samples.tx_off_axis_rad,
        rx_off_axis_rad=reference.samples.rx_off_axis_rad,
        range_rate_km_s=reference.samples.range_rate_km_s,
        earth_clear=reference.samples.earth_clear,
    )
    result = evaluate_pro_link_budget(
        request.profile,
        times_s=geometry.times_s,
        range_km=geometry.range_km,
        elevation_rad=geometry.elevation_rad,
        tx_off_axis_rad=geometry.tx_off_axis_rad,
        rx_off_axis_rad=geometry.rx_off_axis_rad,
        range_rate_km_s=geometry.range_rate_km_s,
        earth_clear=geometry.earth_clear,
        minimum_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
        maximum_range_km=float(station["max_range_km"]),
        resource_limits=resource_limits,
    )
    provenance = {
        "adapter_version": PRO_SCENARIO_ADAPTER_VERSION,
        "scenario_path": str(source.scenario_path),
        "scenario_sha256": source.scenario_sha256,
        "review_path": str(source.review_path),
        "review_sha256": source.review_sha256,
        "normalized_config_sha256": source.normalized_config_sha256,
        "profile_sha256": request.profile.semantic_sha256,
        "geometry_sha256": geometry.semantic_sha256,
        "result_semantic_sha256": result.semantic_sha256,
    }
    return ProScenarioLinkProduct(request, source, geometry, result, provenance)


def evaluate_scenario_rf_coverage(
    scenario_path: str | Path,
    config: ProCommunicationsCoverageConfig,
    *,
    workspace_root: str | Path | None = None,
    resource_limits: ResourceLimits | None = None,
    sample_stride: int = 1,
) -> ProCommunicationsCoverageResult:
    """Evaluate Pro RF coverage directly from one completed scenario object history."""

    if not isinstance(config, ProCommunicationsCoverageConfig):
        raise ValueError("config must be ProCommunicationsCoverageConfig.")
    stride = int(sample_stride)
    if isinstance(sample_stride, (bool, np.bool_)) or stride != sample_stride or stride <= 0:
        raise ValueError("sample_stride must be a positive integer.")
    source = load_completed_scenario_object(
        scenario_path,
        object_id=config.source_asset_id,
        attitude_source_kind=config.attitude_source_kind,
        attitude_provider_id=config.attitude_provider_id,
        workspace_root=workspace_root,
        resource_limits=resource_limits,
    )
    return evaluate_pro_communications_coverage(
        config,
        times_s=source.times_s[::stride],
        positions_eci_km=source.positions_eci_km[::stride],
        attitudes_quat_bn=(None if source.attitudes_quat_bn is None else source.attitudes_quat_bn[::stride]),
        frame_context=source.frame_context,
        resource_limits=resource_limits,
    )


__all__ = [
    "PRO_SCENARIO_ADAPTER_VERSION",
    "CompletedScenarioObjectEvidence",
    "ProScenarioFixedSiteLinkRequest",
    "ProScenarioLinkProduct",
    "evaluate_scenario_fixed_site_link",
    "evaluate_scenario_rf_coverage",
    "load_completed_scenario_object",
]
