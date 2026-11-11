"""Trusted-config Pro communications flagship workflow runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from sim.analysis.coverage_sensitivity import CoverageSensitivityCriteria
from sim.analysis.coverage_tasking import TaskingConstraints
from sim.analysis.directed_link import (
    DirectedLinkConfig,
    LinkTerminal,
    TerminalPattern,
    evaluate_directed_link,
    fixed_wgs84_site_history,
    spacecraft_endpoint_history,
)
from sim.dynamics.orbit.frames import FrameContext
from sim.pro_communications.artifacts import (
    ProStudyArtifacts,
    register_bundle_artifacts,
    write_pro_study_bundle,
)
from sim.pro_communications.availability import (
    evaluate_statistical_aggregate_availability,
    evaluate_statistical_rain_availability,
)
from sim.pro_communications.campaign import (
    LinkGeometryEvidence,
    TradeDimension,
    UncertaintyDimension,
    run_trade_campaign,
    run_uncertainty_campaign,
)
from sim.pro_communications.campaign_convergence import (
    CampaignConvergenceCriteria,
    evaluate_campaign_convergence,
)
from sim.pro_communications.contracts import AtmosphericProfile, LinkBudgetProfile
from sim.pro_communications.coverage import (
    ProCommunicationsCoverageConfig,
    evaluate_pro_communications_coverage,
)
from sim.pro_communications.equipment_library import load_governed_equipment_library
from sim.pro_communications.link_budget import ProLinkBudgetResult, evaluate_pro_link_budget
from sim.pro_communications.network_trades import (
    GroundNetworkAlternative,
    GroundNetworkSite,
    run_ground_network_trade,
    write_design_trade_packet,
)
from sim.pro_communications.p837 import (
    apply_p837_rainfall_rate,
    evaluate_itu_p837_8_rainfall_rate,
    resolve_p837_map_set,
)
from sim.pro_communications.plotting import (
    write_coverage_fraction_plot,
    write_coverage_margin_map,
    write_margin_comparison_plot,
    write_statistical_availability_plot,
)
from sim.pro_communications.scenario_adapter import (
    PRO_SCENARIO_ADAPTER_VERSION,
    load_completed_scenario_object,
)
from sim.pro_communications.scheduling import generate_downlink_opportunities, schedule_downlink_opportunities
from sim.pro_communications.sensitivity import evaluate_pro_coverage_sensitivity

WORKFLOW_SCHEMA_VERSION = "oel.pro-communications-workflow.v0.1"
VALIDATION_MANIFEST_VERSION = "oel.pro-communications-validation-manifest.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_workflow(path: Path, root: Path) -> dict[str, Any]:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Pro communications workflow config must resolve inside the workspace.") from exc
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed Pro communications workflow config.")
    required = {
        "study_id",
        "scenario_path",
        "profile_path",
        "equipment_library_path",
        "object_id",
        "station_id",
        "s_band_profile_id",
        "ka_band_profile_id",
        "coverage_order",
        "coverage_stride",
        "coverage_sensitivity",
        "campaigns",
        "aggregate_availability",
        "schedule",
        "ground_network_alternatives",
        "claims",
        "non_claims",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - {"schema_version"})
    if missing or unknown:
        raise ValueError(f"Workflow fields are incomplete or unknown; missing={missing}, unknown={unknown}.")
    for field_name in ("scenario_path", "profile_path", "equipment_library_path"):
        resolved = (root / str(payload[field_name])).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field_name} must resolve inside the workspace.") from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload[f"_{field_name}"] = resolved
    return payload


def _load_validation_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("manifest_version") != VALIDATION_MANIFEST_VERSION:
        raise ValueError("The supplied validation manifest has an unsupported contract version.")
    if payload.get("status") != "pass":
        raise ValueError("The supplied validation manifest is not passing.")
    declared_semantic_sha256 = payload.get("semantic_sha256")
    if not isinstance(declared_semantic_sha256, str) or len(declared_semantic_sha256) != 64:
        raise ValueError("The supplied validation manifest lacks a semantic SHA-256.")
    identity = dict(payload)
    identity.pop("semantic_sha256")
    actual_semantic_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if actual_semantic_sha256 != declared_semantic_sha256:
        raise ValueError("The supplied validation manifest does not match its semantic SHA-256.")
    components = payload.get("components")
    retained = payload.get("retained_artifacts")
    if not isinstance(components, list) or not isinstance(retained, list):
        raise ValueError("The supplied validation manifest has malformed artifact inventories.")
    return payload, {
        "path": str(path),
        "sha256": _sha256(path),
        "semantic_sha256": declared_semantic_sha256,
        "manifest_version": VALIDATION_MANIFEST_VERSION,
    }


def _public_pattern(profile: LinkBudgetProfile, endpoint: str) -> TerminalPattern:
    antenna = profile.transmitter.antenna if endpoint == "transmitter" else profile.receiver.antenna
    if antenna.kind == "constant":
        return TerminalPattern(kind="constant", gain_dbi=antenna.peak_gain_dbi)
    if antenna.kind == "axisymmetric_hard_cone":
        return TerminalPattern(
            kind="axisymmetric_hard_cone",
            gain_dbi=antenna.peak_gain_dbi,
            half_angle_rad=np.deg2rad(antenna.half_angle_deg),
        )
    return TerminalPattern(
        kind="axisymmetric_hard_cone",
        gain_dbi=antenna.peak_gain_dbi,
        half_angle_rad=np.deg2rad(antenna.off_axis_deg[-1]),
    )


def _free_space_variant(profile: LinkBudgetProfile, suffix: str) -> LinkBudgetProfile:
    return replace(
        profile,
        profile_id=f"{profile.profile_id}-{suffix}",
        atmosphere=AtmosphericProfile(profile_id=f"{profile.atmosphere.profile_id}-{suffix}"),
    )


def _ka_clear_variant(profile: LinkBudgetProfile) -> LinkBudgetProfile:
    atmosphere = replace(
        profile.atmosphere,
        profile_id=f"{profile.atmosphere.profile_id}-gas-cloud",
        rain_model="none",
        rainfall_rate_mm_h=None,
        rain_height_km=None,
        station_latitude_deg=None,
    )
    return replace(profile, profile_id=f"{profile.profile_id}-gas-cloud", atmosphere=atmosphere)


def _reference_geometry(
    profile: LinkBudgetProfile,
    *,
    object_id: str,
    station: Mapping[str, Any],
    scenario_name: str,
    jd_utc: float,
    times: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[LinkGeometryEvidence, np.ndarray]:
    frame_context = FrameContext(jd_utc_start=jd_utc)
    spacecraft = spacecraft_endpoint_history(
        asset_id=object_id,
        state_provider_id=f"review:{scenario_name}:{object_id}",
        times_s=times,
        positions_eci_km=positions,
        velocities_eci_km_s=velocities,
        attitudes_quat_bn=None,
        attitude_source_kind="not_required",
        attitude_provider_id=None,
    )
    site = fixed_wgs84_site_history(
        asset_id=str(station["id"]),
        state_provider_id=f"fixed_wgs84:{station['id']}",
        times_s=times,
        geodetic_latitude_deg=float(station["lat_deg"]),
        longitude_deg=float(station["lon_deg"]),
        ellipsoidal_height_km=float(station["alt_km"]),
        frame_context=frame_context,
    )
    fixed_named_loss = profile.polarization_loss_db + profile.pointing_loss_db + profile.implementation_loss_db
    reference = evaluate_directed_link(
        DirectedLinkConfig(
            analysis_id=f"{profile.profile_id}-public-reference",
            link_id=f"{object_id}-to-{station['id']}",
            tx_terminal=LinkTerminal(
                terminal_id=profile.transmitter.transmitter_id,
                asset_id=object_id,
                parent_frame="body",
                quat_parent_from_terminal=(1.0, 0.0, 0.0, 0.0),
                pattern=_public_pattern(profile, "transmitter"),
            ),
            rx_terminal=LinkTerminal(
                terminal_id=profile.receiver.receiver_id,
                asset_id=str(station["id"]),
                parent_frame="enu",
                quat_parent_from_terminal=(1.0, 0.0, 0.0, 0.0),
                pattern=_public_pattern(profile, "receiver"),
            ),
            carrier_frequency_hz=profile.carrier_frequency_hz,
            tx_power_w=profile.transmitter.power_w,
            data_rate_bps=profile.waveform.data_rate_bps,
            system_noise_temperature_k=profile.receiver.resolved_system_noise_temperature_k,
            required_eb_n0_db=profile.waveform.required_eb_n0_db,
            tx_line_loss_db=profile.transmitter.line_loss_db,
            rx_line_loss_db=profile.receiver.line_loss_db,
            misc_loss_db=profile.legacy_misc_loss_db + fixed_named_loss,
            min_fixed_site_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
            max_range_km=float(station["max_range_km"]),
        ),
        tx_history=spacecraft,
        rx_history=site,
        frame_context=frame_context,
    )
    samples = reference.samples
    geometry = LinkGeometryEvidence(
        source_id=reference.input_evidence_sha256,
        times_s=samples.time_s,
        range_km=samples.range_km,
        elevation_rad=samples.fixed_site_elevation_rad,
        tx_off_axis_rad=samples.tx_off_axis_rad,
        rx_off_axis_rad=samples.rx_off_axis_rad,
        range_rate_km_s=samples.range_rate_km_s,
        earth_clear=samples.earth_clear,
    )
    return geometry, reference.samples.margin_db


def _evaluate(
    profile: LinkBudgetProfile, geometry: LinkGeometryEvidence, station: Mapping[str, Any]
) -> ProLinkBudgetResult:
    return evaluate_pro_link_budget(
        profile,
        times_s=geometry.times_s,
        range_km=geometry.range_km,
        elevation_rad=geometry.elevation_rad,
        tx_off_axis_rad=geometry.tx_off_axis_rad,
        rx_off_axis_rad=geometry.rx_off_axis_rad,
        range_rate_km_s=geometry.range_rate_km_s,
        earth_clear=geometry.earth_clear,
        minimum_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
        maximum_range_km=float(station["max_range_km"]),
    )


def run_flagship_workflow(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    validation_manifest: str | Path | None = None,
    p837_maps: str | Path | None = None,
) -> ProStudyArtifacts:
    root = Path(__file__).resolve().parents[2]
    config_source = Path(config_path).expanduser().resolve()
    config = _load_workflow(config_source, root)
    scenario_path = config["_scenario_path"]
    profile_path = config["_profile_path"]
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    library = load_governed_equipment_library(config["_equipment_library_path"])
    if library.profile_file != profile_path:
        raise ValueError("The governed equipment library does not bind the configured profile_path.")
    profiles = {key: value.profile for key, value in library.entries.items()}
    s_profile = profiles[str(config["s_band_profile_id"])]
    ka_profile = profiles[str(config["ka_band_profile_id"])]
    station = next(
        (value for value in scenario["ground_stations"] if value["id"] == config["station_id"]),
        None,
    )
    if station is None:
        raise ValueError("Configured station_id is absent from the scenario.")
    p837_result = None
    if p837_maps is not None:
        p837_map_set = resolve_p837_map_set(p837_maps)
        p837_result = evaluate_itu_p837_8_rainfall_rate(
            p837_map_set,
            exceedance_percentage=(float(ka_profile.atmosphere.rain_exceedance_percentage),),
            latitude_deg=float(station["lat_deg"]),
            longitude_deg=float(station["lon_deg"]),
        )
        ka_profile = replace(
            ka_profile,
            atmosphere=apply_p837_rainfall_rate(
                ka_profile.atmosphere,
                p837_result,
                exceedance_percentage=float(ka_profile.atmosphere.rain_exceedance_percentage),
            ),
        )
    source = load_completed_scenario_object(
        scenario_path,
        object_id=str(config["object_id"]),
        attitude_source_kind="not_required",
        workspace_root=root,
    )
    review_source = source.review_path
    times = source.times_s
    positions = source.positions_eci_km
    velocities = source.velocities_eci_km_s
    jd_utc = float(scenario["simulator"]["initial_jd_utc"])
    reference_profile = _free_space_variant(s_profile, "free-space")
    geometry, public_reference_margin = _reference_geometry(
        reference_profile,
        object_id=str(config["object_id"]),
        station=station,
        scenario_name=str(scenario["scenario_name"]),
        jd_utc=jd_utc,
        times=times,
        positions=positions,
        velocities=velocities,
    )
    s_free = _evaluate(reference_profile, geometry, station)
    legacy_difference = float(np.max(np.abs(s_free.samples.margin_db - public_reference_margin)))
    if legacy_difference > 1.0e-12:
        raise AssertionError(f"Legacy free-space parity failed by {legacy_difference} dB.")
    s_gas = _evaluate(s_profile, geometry, station)
    ka_free_profile = _free_space_variant(ka_profile, "free-space")
    ka_clear_profile = _ka_clear_variant(ka_profile)
    ka_free = _evaluate(ka_free_profile, geometry, station)
    ka_clear = _evaluate(ka_clear_profile, geometry, station)
    ka_weather = _evaluate(ka_profile, geometry, station)

    stride = int(config["coverage_stride"])
    if stride <= 0:
        raise ValueError("coverage_stride must be positive.")
    coverage_slice = slice(None, None, stride)
    frame_context = FrameContext(jd_utc_start=jd_utc)
    coverages = {
        "s-band-gas": evaluate_pro_communications_coverage(
            ProCommunicationsCoverageConfig(
                analysis_id="colorado-springs-s-band-rf-coverage",
                service_id="s-band-declared-rate-closure",
                source_asset_id=str(config["object_id"]),
                state_provider_id=geometry.source_id,
                attitude_source_kind="not_required",
                attitude_provider_id=None,
                quat_body_from_source_terminal=(1.0, 0.0, 0.0, 0.0),
                direction="spacecraft_to_earth",
                order=int(config["coverage_order"]),
                link_profile=s_profile,
                minimum_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
                maximum_range_km=float(station["max_range_km"]),
            ),
            times_s=times[coverage_slice],
            positions_eci_km=positions[coverage_slice],
            attitudes_quat_bn=None,
            frame_context=frame_context,
        ),
        "ka-gas-cloud": evaluate_pro_communications_coverage(
            ProCommunicationsCoverageConfig(
                analysis_id="colorado-springs-ka-clear-rf-coverage",
                service_id="ka-declared-rate-clear-sky-closure",
                source_asset_id=str(config["object_id"]),
                state_provider_id=geometry.source_id,
                attitude_source_kind="not_required",
                attitude_provider_id=None,
                quat_body_from_source_terminal=(1.0, 0.0, 0.0, 0.0),
                direction="spacecraft_to_earth",
                order=int(config["coverage_order"]),
                link_profile=ka_clear_profile,
                minimum_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
                maximum_range_km=float(station["max_range_km"]),
            ),
            times_s=times[coverage_slice],
            positions_eci_km=positions[coverage_slice],
            attitudes_quat_bn=None,
            frame_context=frame_context,
        ),
        "ka-weather": evaluate_pro_communications_coverage(
            ProCommunicationsCoverageConfig(
                analysis_id="colorado-springs-ka-rf-coverage",
                service_id="ka-declared-rate-weather-sensitive-closure",
                source_asset_id=str(config["object_id"]),
                state_provider_id=geometry.source_id,
                attitude_source_kind="not_required",
                attitude_provider_id=None,
                quat_body_from_source_terminal=(1.0, 0.0, 0.0, 0.0),
                direction="spacecraft_to_earth",
                order=int(config["coverage_order"]),
                link_profile=ka_profile,
                minimum_elevation_rad=np.deg2rad(float(station["min_elevation_deg"])),
                maximum_range_km=float(station["max_range_km"]),
            ),
            times_s=times[coverage_slice],
            positions_eci_km=positions[coverage_slice],
            attitudes_quat_bn=None,
            frame_context=frame_context,
        ),
    }
    sensitivity_config = config["coverage_sensitivity"]
    cadence_stride = int(sensitivity_config["cadence_refined_stride"])
    resolution_order = int(sensitivity_config["resolution_refined_order"])
    if cadence_stride <= 0 or stride % cadence_stride != 0 or cadence_stride >= stride:
        raise ValueError("cadence_refined_stride must be a positive proper divisor of coverage_stride.")
    if resolution_order != int(config["coverage_order"]) + 1:
        raise ValueError("resolution_refined_order must be exactly one order above coverage_order.")
    s_cadence_refined = evaluate_pro_communications_coverage(
        replace(
            coverages["s-band-gas"].config,
            analysis_id="colorado-springs-s-band-rf-coverage-cadence-refined",
        ),
        times_s=times[::cadence_stride],
        positions_eci_km=positions[::cadence_stride],
        attitudes_quat_bn=None,
        frame_context=frame_context,
    )
    s_resolution_refined = evaluate_pro_communications_coverage(
        replace(
            coverages["s-band-gas"].config,
            analysis_id="colorado-springs-s-band-rf-coverage-resolution-refined",
            order=resolution_order,
        ),
        times_s=times[coverage_slice],
        positions_eci_km=positions[coverage_slice],
        attitudes_quat_bn=None,
        frame_context=frame_context,
    )
    sensitivity_criteria = CoverageSensitivityCriteria(
        maximum_mean_fraction_absolute_delta=float(sensitivity_config["maximum_mean_fraction_absolute_delta"]),
        maximum_ever_fraction_absolute_delta=float(sensitivity_config["maximum_ever_fraction_absolute_delta"]),
    )
    coverage_sensitivity = {
        "s-band-cadence": evaluate_pro_coverage_sensitivity(
            comparison_id="colorado-springs-s-band-cadence",
            comparison_kind="cadence",
            baseline=coverages["s-band-gas"],
            refined=s_cadence_refined,
            criteria=sensitivity_criteria,
        ),
        "s-band-resolution": evaluate_pro_coverage_sensitivity(
            comparison_id="colorado-springs-s-band-resolution",
            comparison_kind="resolution",
            baseline=coverages["s-band-gas"],
            refined=s_resolution_refined,
            criteria=sensitivity_criteria,
        ),
    }
    evidence_coverages = {
        **coverages,
        "s-band-gas-cadence-refined": s_cadence_refined,
        "s-band-gas-resolution-refined": s_resolution_refined,
    }

    schedule_config = config["schedule"]
    opportunities = generate_downlink_opportunities(
        s_gas,
        asset_id=str(config["object_id"]),
        energy_cost_w=float(schedule_config["downlink_power_w"]),
    )
    schedule = schedule_downlink_opportunities(
        opportunities,
        analysis_id="colorado-springs-s-band-contact-plan",
        asset_id=str(config["object_id"]),
        constraints=TaskingConstraints(
            horizon_start_s=float(times[0]),
            horizon_end_s=float(times[-1]),
            maximum_slew_rate_rad_s=None,
            settling_time_s=0.0,
            maximum_payload_duty_cycle=1.0,
            storage_capacity_bytes=float(schedule_config["storage_capacity_bytes"]),
            initial_storage_bytes=float(schedule_config["initial_storage_bytes"]),
            energy_budget_wh=float(schedule_config["energy_budget_wh"]),
        ),
    )
    campaigns_config = config["campaigns"]
    campaigns = {
        "s-band-receiver-temperature": run_trade_campaign(
            "s-band-receiver-temperature",
            s_profile,
            geometry,
            (
                TradeDimension(
                    "receiver.system_noise_temperature_k",
                    tuple(campaigns_config["s_band_receiver_temperature_k"]),
                ),
            ),
        ),
        "ka-pointing-loss": run_trade_campaign(
            "ka-pointing-loss",
            ka_profile,
            geometry,
            (TradeDimension("pointing_loss_db", tuple(campaigns_config["ka_pointing_loss_db"])),),
        ),
        "ka-rain-percentage": run_trade_campaign(
            "ka-rain-percentage",
            ka_profile,
            geometry,
            (
                TradeDimension(
                    "atmosphere.rain_exceedance_percentage",
                    tuple(campaigns_config["ka_rain_exceedance_percentage"]),
                ),
            ),
        ),
        "s-band-seeded-uncertainty": run_uncertainty_campaign(
            "s-band-seeded-uncertainty",
            s_profile,
            geometry,
            (
                UncertaintyDimension("transmitter.power_w", 10.0, 0.5),
                UncertaintyDimension("receiver.system_noise_temperature_k", 500.0, 25.0),
            ),
            case_count=32,
            seed=20260823,
            correlation_matrix=np.asarray([[1.0, -0.2], [-0.2, 1.0]]),
        ),
    }
    campaign_convergence = {
        "s-band-seeded-uncertainty": evaluate_campaign_convergence(
            campaigns["s-band-seeded-uncertainty"],
            CampaignConvergenceCriteria(
                maximum_mean_margin_absolute_delta_db=1.0,
                maximum_p10_margin_absolute_delta_db=1.5,
                maximum_mean_delivery_relative_delta=0.10,
                maximum_delivery_probability_absolute_delta=0.10,
                minimum_cases_per_half=16,
            ),
        )
    }
    scenario_stations = {str(value["id"]): value for value in scenario["ground_stations"]}
    site_geometry: dict[str, LinkGeometryEvidence] = {str(station["id"]): geometry}
    site_definitions: dict[str, Mapping[str, Any]] = {str(station["id"]): station}
    ground_alternatives: list[GroundNetworkAlternative] = []
    for alternative_record in config["ground_network_alternatives"]:
        if not isinstance(alternative_record, dict) or set(alternative_record) != {"alternative_id", "sites"}:
            raise ValueError("Every ground-network alternative requires exactly alternative_id and sites.")
        sites: dict[str, GroundNetworkSite] = {}
        for site_record in alternative_record["sites"]:
            if not isinstance(site_record, dict):
                raise ValueError("Ground-network sites must be objects.")
            if set(site_record) == {"scenario_station_id"}:
                site_id = str(site_record["scenario_station_id"])
                site_definition = scenario_stations.get(site_id)
                if site_definition is None:
                    raise ValueError(f"Unknown scenario ground station {site_id!r}.")
            else:
                expected = {"id", "lat_deg", "lon_deg", "alt_km", "min_elevation_deg", "max_range_km"}
                if set(site_record) != expected:
                    raise ValueError("Explicit ground-network sites have incomplete or unknown fields.")
                site_definition = site_record
                site_id = str(site_definition["id"])
            if site_id not in site_geometry:
                site_geometry[site_id], _ = _reference_geometry(
                    s_profile,
                    object_id=str(config["object_id"]),
                    station=site_definition,
                    scenario_name=str(scenario["scenario_name"]),
                    jd_utc=jd_utc,
                    times=times,
                    positions=positions,
                    velocities=velocities,
                )
            site_definitions[site_id] = site_definition
            sites[site_id] = GroundNetworkSite(
                geometry=site_geometry[site_id],
                minimum_elevation_rad=np.deg2rad(float(site_definition["min_elevation_deg"])),
                maximum_range_km=float(site_definition["max_range_km"]),
            )
        ground_alternatives.append(GroundNetworkAlternative(str(alternative_record["alternative_id"]), sites))
    design_trades = {
        "s-band-ground-network": run_ground_network_trade(
            "colorado-springs-s-band-ground-network",
            s_profile,
            tuple(ground_alternatives),
        )
    }
    network_opportunities = tuple(
        opportunity
        for site_id, site_link_geometry in sorted(site_geometry.items())
        for opportunity in generate_downlink_opportunities(
            _evaluate(s_profile, site_link_geometry, site_definitions[site_id]),
            asset_id=str(config["object_id"]),
            energy_cost_w=float(schedule_config["downlink_power_w"]),
            opportunity_namespace=f"s-band:{site_id}",
            target_id=site_id,
        )
    )
    network_schedule = schedule_downlink_opportunities(
        network_opportunities,
        analysis_id="illustrative-two-site-s-band-contact-plan",
        asset_id=str(config["object_id"]),
        constraints=TaskingConstraints(
            horizon_start_s=float(times[0]),
            horizon_end_s=float(times[-1]),
            maximum_slew_rate_rad_s=None,
            settling_time_s=0.0,
            maximum_payload_duty_cycle=float(schedule_config["network_maximum_payload_duty_cycle"]),
            storage_capacity_bytes=float(schedule_config["storage_capacity_bytes"]),
            initial_storage_bytes=float(schedule_config["initial_storage_bytes"]),
            energy_budget_wh=float(schedule_config["network_energy_budget_wh"]),
        ),
    )
    schedules = {
        "s-band-contact-plan": schedule,
        "s-band-two-site-constrained-plan": network_schedule,
    }
    peak_index = int(np.argmax(geometry.elevation_rad))
    statistical_rain = evaluate_statistical_rain_availability(
        ka_profile.atmosphere,
        carrier_frequency_hz=ka_profile.carrier_frequency_hz,
        elevation_rad=float(geometry.elevation_rad[peak_index]),
        clear_sky_margin_db=float(ka_clear.samples.margin_db[peak_index]),
    )
    aggregate_config = config["aggregate_availability"]
    if not isinstance(aggregate_config, dict) or set(aggregate_config) != {
        "median_wet_refractivity_n_units",
        "antenna_diameter_m",
        "antenna_efficiency",
    }:
        raise ValueError("aggregate_availability has incomplete or unknown fields.")
    statistical_aggregate = evaluate_statistical_aggregate_availability(
        ka_profile.atmosphere,
        carrier_frequency_hz=ka_profile.carrier_frequency_hz,
        elevation_rad=float(geometry.elevation_rad[peak_index]),
        clear_path_margin_db=float(ka_free.samples.margin_db[peak_index]),
        median_wet_refractivity_n_units=float(aggregate_config["median_wet_refractivity_n_units"]),
        antenna_diameter_m=float(aggregate_config["antenna_diameter_m"]),
        antenna_efficiency=float(aggregate_config["antenna_efficiency"]),
    )
    statistical_products = {
        "ka-peak-elevation-rain": statistical_rain,
        "ka-peak-elevation-aggregate": statistical_aggregate,
    }
    validation_source: dict[str, Any] | None = None
    validation_payload: dict[str, Any] | None = None
    if validation_manifest is not None:
        validation_path = Path(validation_manifest).expanduser().resolve()
        validation_payload, validation_source = _load_validation_manifest(validation_path)
    provenance = {
        "workflow_config": {"path": str(config_source.relative_to(root)), "sha256": _sha256(config_source)},
        "scenario": {"path": str(scenario_path.relative_to(root)), "sha256": _sha256(scenario_path)},
        "profiles": {"path": str(profile_path.relative_to(root)), "sha256": _sha256(profile_path)},
        "equipment_library": {
            "path": str(library.source_path.relative_to(root)),
            "sha256": library.source_sha256,
            "semantic_sha256": library.semantic_sha256,
            "library_id": library.library_id,
            "library_version": library.library_version,
            "data_classification": library.data_classification,
            "entry_dispositions": {
                key: {
                    "calibration_status": value.calibration_status,
                    "use_disposition": value.use_disposition,
                }
                for key, value in library.entries.items()
            },
        },
        "review_evidence": {
            "path": str(review_source.relative_to(root)),
            "sha256": source.review_sha256,
            "normalized_config_sha256": source.normalized_config_sha256,
        },
        "scenario_adapter_version": PRO_SCENARIO_ADAPTER_VERSION,
        "scenario_adapter_attitude_source_kind": source.attitude_source_kind,
        "review_sample_count": int(times.size),
        "coverage_stride": stride,
        "legacy_free_space_maximum_margin_difference_db": legacy_difference,
        "validation_manifest": validation_source,
        "p837_8_climatology": (
            None
            if p837_result is None
            else {
                "semantic_sha256": p837_result.semantic_sha256,
                "latitude_deg": p837_result.latitude_deg,
                "longitude_deg": p837_result.longitude_deg,
                "exceedance_percentage": p837_result.exceedance_percentage,
                "annual_rainfall_rate_mm_h": p837_result.annual_rainfall_rate_mm_h,
                "map_resolution_deg": p837_result.map_resolution_deg,
                "map_hashes": p837_result.map_hashes,
            }
        ),
    }
    links = {
        "s-band-free-space": s_free,
        "s-band-p676-gas": s_gas,
        "ka-free-space": ka_free,
        "ka-gas-cloud": ka_clear,
        "ka-gas-cloud-rain": ka_weather,
    }
    bundle = write_pro_study_bundle(
        output_dir,
        study_id=str(config["study_id"]),
        links=links,
        coverages=evidence_coverages,
        statistical_availability=statistical_products,
        schedules=schedules,
        campaigns=campaigns,
        campaign_convergence=campaign_convergence,
        coverage_sensitivity=coverage_sensitivity,
        design_trades=design_trades,
        study_provenance=provenance,
        study_claims=tuple(config["claims"]),
        study_non_claims=tuple(config["non_claims"]),
    )
    plots_dir = bundle.output_dir / "plots"
    plots_dir.mkdir()
    plots = {
        "s-band-margin-comparison": write_margin_comparison_plot(
            plots_dir / "s_band_margin_comparison.png",
            study_id=str(config["study_id"]),
            title="Colorado Springs S-Band Downlink Margin",
            links={"free space": s_free, "P.676-13 gas": s_gas},
        ),
        "ka-margin-comparison": write_margin_comparison_plot(
            plots_dir / "ka_margin_comparison.png",
            study_id=str(config["study_id"]),
            title="Colorado Springs Ka-Band Weather-Sensitive Margin",
            links={"free space": ka_free, "gas + cloud": ka_clear, "gas + cloud + rain": ka_weather},
        ),
        "rf-coverage-fraction": write_coverage_fraction_plot(
            plots_dir / "rf_coverage_fraction.png",
            study_id=str(config["study_id"]),
            coverages=coverages,
        ),
        "ka-coverage-margin-map": write_coverage_margin_map(
            plots_dir / "ka_coverage_margin_map.png",
            study_id=str(config["study_id"]),
            title="Ka-Band Clear-Sky Best Sampled Whole-Earth Cell Margin",
            coverage=coverages["ka-gas-cloud"],
        ),
        "ka-rain-exceedance": write_statistical_availability_plot(
            plots_dir / "ka_rain_exceedance.png",
            study_id=str(config["study_id"]),
            result=statistical_rain,
        ),
        "ka-aggregate-exceedance": write_statistical_availability_plot(
            plots_dir / "ka_aggregate_exceedance.png",
            study_id=str(config["study_id"]),
            result=statistical_aggregate,
        ),
    }
    receipt_path = bundle.output_dir / "workflow_receipt.json"
    receipt = {
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "status": "complete",
        "provenance": provenance,
        "source_product_semantic_hashes": {
            "links": {key: value.semantic_sha256 for key, value in links.items()},
            "coverages": {key: value.semantic_sha256 for key, value in evidence_coverages.items()},
            "schedules": {key: value.semantic_sha256 for key, value in schedules.items()},
            "campaigns": {key: value.semantic_sha256 for key, value in campaigns.items()},
            "campaign_convergence": {key: value.semantic_sha256 for key, value in campaign_convergence.items()},
            "statistical_availability": {key: value.semantic_sha256 for key, value in statistical_products.items()},
            "coverage_sensitivity": {key: value.semantic_sha256 for key, value in coverage_sensitivity.items()},
            "design_trades": {key: value.semantic_sha256 for key, value in design_trades.items()},
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    extra_artifacts: dict[str, Path] = {**plots, "workflow-receipt": receipt_path}
    trade_packet = write_design_trade_packet(
        design_trades["s-band-ground-network"],
        bundle.output_dir / "s_band_ground_network_trade.json",
    )
    extra_artifacts["s-band-ground-network-trade"] = trade_packet
    if validation_manifest is not None:
        validation_copy = bundle.output_dir / "validation_manifest.json"
        shutil.copyfile(Path(validation_manifest).expanduser().resolve(), validation_copy)
        extra_artifacts["validation-manifest"] = validation_copy
        if validation_payload is None:
            raise RuntimeError("Validation payload was unexpectedly unavailable.")
        validation_dir = bundle.output_dir / "validation"
        validation_dir.mkdir()
        sources = [
            (f"component-{value['component_id']}", Path(value["source_path"]), value["sha256"])
            for value in validation_payload.get("components", [])
        ]
        sources.extend(
            (f"raw-{value['artifact_id']}", Path(value["source_path"]), value["sha256"])
            for value in validation_payload.get("retained_artifacts", [])
        )
        for artifact_id, source, expected_sha256 in sources:
            resolved_source = source.expanduser().resolve()
            if _sha256(resolved_source) != expected_sha256:
                raise ValueError(f"Validation artifact changed after manifest assembly: {resolved_source}")
            suffix = resolved_source.suffix or ".bin"
            destination = validation_dir / f"{artifact_id}{suffix}"
            shutil.copyfile(resolved_source, destination)
            extra_artifacts[f"validation-{artifact_id}"] = destination
    register_bundle_artifacts(bundle, extra_artifacts)
    return bundle


def main() -> int:
    from sim.pro_features import FEATURE_PRO_COMMUNICATIONS, require_pro_feature

    require_pro_feature(FEATURE_PRO_COMMUNICATIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument(
        "--p837-maps",
        type=Path,
        help="Extracted official ITU-R P.837-8 Annex 1 package directory.",
    )
    args = parser.parse_args()
    artifacts = run_flagship_workflow(
        args.config,
        args.output,
        validation_manifest=args.validation_manifest,
        p837_maps=args.p837_maps,
    )
    print(artifacts.manifest_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
