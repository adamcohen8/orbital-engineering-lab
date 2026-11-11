"""Durable, queryable evidence bundles for Pro communications studies."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.analysis.coverage_aggregation import ConstellationCoverageResult
from sim.analysis.coverage_sensitivity import CoverageSensitivityResult
from sim.pro_communications.availability import (
    StatisticalAggregateAvailabilityResult,
    StatisticalRainAvailabilityResult,
)
from sim.pro_communications.campaign import ProCampaignResult
from sim.pro_communications.campaign_convergence import CampaignConvergenceResult
from sim.pro_communications.coverage import ProCommunicationsCoverageResult
from sim.pro_communications.link_budget import ProLinkBudgetResult
from sim.pro_communications.network_trades import ProDesignTradeResult
from sim.pro_communications.scheduling import ProScheduleResult

PRO_EVIDENCE_MANIFEST_VERSION = "oel.pro-communications-evidence.v0.4"


@dataclass(frozen=True)
class ProStudyArtifacts:
    output_dir: Path
    manifest_json: Path
    summary_json: Path
    review_sqlite: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sqlite_optional_real(value: Any) -> float | None:
    """Encode missing floating-point evidence as SQL NULL.

    NaN is how the RF evaluator represents a derived quantity outside a
    declared model's applicability domain.  SQLite binds NaN as NULL, so do
    that conversion explicitly before insertion and keep the evidence
    disposition columns available to explain why the value is missing.  A
    signed infinity remains a numeric sentinel where the source contract uses
    it (for example, zero external interference in dB units).
    """

    normalized = float(value)
    return None if np.isnan(normalized) else normalized


def _sqlite_missing_real(value: Any) -> float | None:
    """Encode a non-finite derived summary as SQL NULL."""

    normalized = float(value)
    return None if not np.isfinite(normalized) else normalized


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE study_metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE pro_link_samples (
            analysis_key TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            time_s REAL NOT NULL,
            range_km REAL NOT NULL,
            elevation_deg REAL NOT NULL,
            tx_off_axis_deg REAL NOT NULL,
            rx_off_axis_deg REAL NOT NULL,
            terrain_clear INTEGER NOT NULL,
            atmospheric_applicability_pass INTEGER NOT NULL,
            free_space_path_loss_db REAL NOT NULL,
            gaseous_loss_db REAL,
            rain_loss_db REAL,
            cloud_loss_db REAL,
            received_power_dbw REAL,
            thermal_noise_density_dbw_hz REAL NOT NULL,
            external_interference_density_dbw_hz REAL,
            effective_noise_density_dbw_hz REAL,
            cn0_db_hz REAL,
            eb_n0_db REAL,
            required_eb_n0_db REAL NOT NULL,
            margin_db REAL,
            acquisition_threshold_pass INTEGER NOT NULL,
            tracking_threshold_pass INTEGER NOT NULL,
            receiver_locked INTEGER NOT NULL,
            selected_data_rate_bps REAL,
            selected_mode_id TEXT NOT NULL,
            available INTEGER NOT NULL,
            primary_reason TEXT NOT NULL,
            doppler_shift_hz REAL NOT NULL,
            PRIMARY KEY (analysis_key, sample_index)
        );
        CREATE TABLE pro_link_windows (
            analysis_key TEXT NOT NULL,
            interval_index INTEGER NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            duration_s REAL NOT NULL,
            start_censored INTEGER NOT NULL,
            end_censored INTEGER NOT NULL,
            acquisition_reason TEXT NOT NULL,
            loss_reason TEXT NOT NULL,
            PRIMARY KEY (analysis_key, interval_index)
        );
        CREATE TABLE pro_coverage_samples (
            analysis_key TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            time_s REAL NOT NULL,
            covered_cell_count INTEGER NOT NULL,
            covered_fraction REAL NOT NULL,
            minimum_margin_db REAL,
            maximum_margin_db REAL,
            reason_count_json TEXT NOT NULL,
            PRIMARY KEY (analysis_key, sample_index)
        );
        CREATE TABLE pro_coverage_cells (
            analysis_key TEXT NOT NULL,
            cell_index INTEGER NOT NULL,
            latitude_deg REAL NOT NULL,
            longitude_deg REAL NOT NULL,
            dwell_s REAL NOT NULL,
            interval_count INTEGER NOT NULL,
            acquisition_count INTEGER NOT NULL,
            max_complete_revisit_gap_s REAL,
            prefix_boundary_gap_s REAL,
            suffix_boundary_gap_s REAL,
            start_censored INTEGER NOT NULL,
            end_censored INTEGER NOT NULL,
            best_margin_db REAL,
            worst_margin_db REAL,
            PRIMARY KEY (analysis_key, cell_index)
        );
        CREATE TABLE pro_statistical_availability (
            analysis_key TEXT NOT NULL,
            curve_index INTEGER NOT NULL,
            exceedance_percentage REAL NOT NULL,
            rain_attenuation_db REAL NOT NULL,
            cloud_attenuation_db REAL NOT NULL,
            gaseous_attenuation_db REAL NOT NULL,
            scintillation_fade_depth_db REAL NOT NULL,
            total_attenuation_db REAL NOT NULL,
            disposition TEXT NOT NULL,
            product_kind TEXT NOT NULL,
            PRIMARY KEY (analysis_key, curve_index)
        );
        CREATE TABLE pro_schedule (
            analysis_key TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            opportunity_id TEXT NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            estimated_data_bits REAL NOT NULL,
            storage_after_bytes REAL NOT NULL,
            cumulative_energy_wh REAL NOT NULL,
            source_product_sha256 TEXT NOT NULL,
            PRIMARY KEY (analysis_key, sequence)
        );
        CREATE TABLE pro_schedule_rejections (
            analysis_key TEXT NOT NULL,
            opportunity_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (analysis_key, opportunity_id)
        );
        CREATE TABLE pro_campaign_cases (
            analysis_key TEXT NOT NULL,
            case_id TEXT NOT NULL,
            parameter_values_json TEXT NOT NULL,
            profile_sha256 TEXT NOT NULL,
            link_semantic_sha256 TEXT NOT NULL,
            minimum_margin_db REAL NOT NULL,
            maximum_margin_db REAL NOT NULL,
            available_duration_s REAL NOT NULL,
            estimated_delivered_data_bits REAL NOT NULL,
            PRIMARY KEY (analysis_key, case_id)
        );
        CREATE TABLE pro_campaign_convergence (
            analysis_key TEXT PRIMARY KEY,
            campaign_semantic_sha256 TEXT NOT NULL,
            passed INTEGER NOT NULL,
            disposition TEXT NOT NULL,
            criteria_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            semantic_sha256 TEXT NOT NULL
        );
        CREATE TABLE pro_constellation_samples (
            analysis_key TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            time_s REAL NOT NULL,
            qualified_cell_count INTEGER NOT NULL,
            qualified_fraction REAL NOT NULL,
            maximum_multiplicity INTEGER NOT NULL,
            active_asset_count INTEGER NOT NULL,
            PRIMARY KEY (analysis_key, sample_index)
        );
        CREATE TABLE pro_constellation_cells (
            analysis_key TEXT NOT NULL,
            cell_index INTEGER NOT NULL,
            mean_multiplicity REAL NOT NULL,
            maximum_multiplicity INTEGER NOT NULL,
            dwell_s REAL NOT NULL,
            interval_count INTEGER NOT NULL,
            PRIMARY KEY (analysis_key, cell_index)
        );
        CREATE TABLE pro_coverage_sensitivity (
            analysis_key TEXT PRIMARY KEY,
            comparison_kind TEXT NOT NULL,
            baseline_semantic_sha256 TEXT NOT NULL,
            refined_semantic_sha256 TEXT NOT NULL,
            baseline_order INTEGER NOT NULL,
            refined_order INTEGER NOT NULL,
            baseline_sample_count INTEGER NOT NULL,
            refined_sample_count INTEGER NOT NULL,
            passed INTEGER NOT NULL,
            disposition TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            semantic_sha256 TEXT NOT NULL
        );
        CREATE TABLE pro_design_trade_cases (
            analysis_key TEXT NOT NULL,
            rank INTEGER NOT NULL,
            alternative_id TEXT NOT NULL,
            source_semantic_sha256_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            recommended INTEGER NOT NULL,
            PRIMARY KEY (analysis_key, rank)
        );
        """
    )


def _insert_link(connection: sqlite3.Connection, key: str, result: ProLinkBudgetResult) -> None:
    samples = result.samples
    connection.executemany(
        """
        INSERT INTO pro_link_samples (
            analysis_key, sample_index, time_s, range_km, elevation_deg,
            tx_off_axis_deg, rx_off_axis_deg, terrain_clear,
            atmospheric_applicability_pass, free_space_path_loss_db,
            gaseous_loss_db, rain_loss_db, cloud_loss_db, received_power_dbw,
            thermal_noise_density_dbw_hz, external_interference_density_dbw_hz,
            effective_noise_density_dbw_hz, cn0_db_hz, eb_n0_db,
            required_eb_n0_db, margin_db, acquisition_threshold_pass,
            tracking_threshold_pass, receiver_locked, selected_data_rate_bps,
            selected_mode_id, available, primary_reason, doppler_shift_hz
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            (
                key,
                index,
                float(samples.time_s[index]),
                float(samples.range_km[index]),
                float(np.rad2deg(samples.elevation_rad[index])),
                float(np.rad2deg(samples.tx_off_axis_rad[index])),
                float(np.rad2deg(samples.rx_off_axis_rad[index])),
                int(samples.terrain_clear[index]),
                int(samples.atmospheric_applicability_pass[index]),
                float(samples.free_space_path_loss_db[index]),
                _sqlite_optional_real(samples.gaseous_loss_db[index]),
                _sqlite_optional_real(samples.rain_loss_db[index]),
                _sqlite_optional_real(samples.cloud_loss_db[index]),
                _sqlite_optional_real(samples.received_power_dbw[index]),
                float(samples.noise_density_dbw_hz[index]),
                _sqlite_optional_real(samples.external_interference_density_dbw_hz[index]),
                _sqlite_optional_real(samples.effective_noise_density_dbw_hz[index]),
                _sqlite_optional_real(samples.cn0_db_hz[index]),
                _sqlite_optional_real(samples.eb_n0_db[index]),
                float(samples.required_eb_n0_db[index]),
                _sqlite_optional_real(samples.margin_db[index]),
                int(samples.acquisition_threshold_pass[index]),
                int(samples.tracking_threshold_pass[index]),
                int(samples.receiver_locked[index]),
                _sqlite_optional_real(samples.selected_data_rate_bps[index]),
                samples.selected_mode_id[index],
                int(samples.available[index]),
                samples.primary_reason[index],
                float(samples.doppler_shift_hz[index]),
            )
            for index in range(samples.time_s.size)
        ),
    )
    connection.executemany(
        "INSERT INTO pro_link_windows VALUES (?,?,?,?,?,?,?,?,?)",
        (
            (
                key,
                interval.interval_index,
                interval.start_s,
                interval.end_s,
                interval.duration_s,
                int(interval.start_censored),
                int(interval.end_censored),
                interval.acquisition_reason,
                interval.loss_reason,
            )
            for interval in result.intervals
        ),
    )


def _insert_coverage(connection: sqlite3.Connection, key: str, result: ProCommunicationsCoverageResult) -> None:
    reason_names = result.summary["reason_names"]
    connection.executemany(
        "INSERT INTO pro_coverage_samples VALUES (?,?,?,?,?,?,?,?)",
        (
            (
                key,
                index,
                float(result.times_s[index]),
                int(result.covered_cell_count[index]),
                float(result.instantaneous_covered_fraction[index]),
                _sqlite_missing_real(result.sample_margin_min_db[index]),
                _sqlite_missing_real(result.sample_margin_max_db[index]),
                json.dumps(
                    {
                        str(reason): int(count)
                        for reason, count in zip(reason_names, result.primary_reason_count[index])
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for index in range(result.times_s.size)
        ),
    )
    metrics = result.cell_metrics
    connection.executemany(
        "INSERT INTO pro_coverage_cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            (
                key,
                int(metrics.cell_index[index]),
                float(result.cell_geodetic_latitude_deg[index]),
                float(result.cell_longitude_deg[index]),
                float(metrics.dwell_s[index]),
                int(metrics.interval_count[index]),
                int(metrics.observed_acquisition_count[index]),
                None
                if not np.isfinite(metrics.max_complete_revisit_gap_s[index])
                else float(metrics.max_complete_revisit_gap_s[index]),
                None
                if not np.isfinite(metrics.prefix_boundary_gap_s[index])
                else float(metrics.prefix_boundary_gap_s[index]),
                None
                if not np.isfinite(metrics.suffix_boundary_gap_s[index])
                else float(metrics.suffix_boundary_gap_s[index]),
                int(metrics.start_censored[index]),
                int(metrics.end_censored[index]),
                None
                if not np.isfinite(result.cell_best_margin_db[index])
                else float(result.cell_best_margin_db[index]),
                None
                if not np.isfinite(result.cell_worst_margin_db[index])
                else float(result.cell_worst_margin_db[index]),
            )
            for index in range(metrics.cell_index.size)
        ),
    )


def _insert_availability(
    connection: sqlite3.Connection,
    key: str,
    result: StatisticalRainAvailabilityResult | StatisticalAggregateAvailabilityResult,
) -> None:
    aggregate = isinstance(result, StatisticalAggregateAvailabilityResult)
    zeros = np.zeros(result.exceedance_percentage.shape)
    cloud = result.cloud_attenuation_db if aggregate else zeros
    gas = result.gaseous_attenuation_db if aggregate else zeros
    scintillation = result.scintillation_fade_depth_db if aggregate else zeros
    total = result.total_attenuation_db if aggregate else result.rain_attenuation_db
    connection.executemany(
        "INSERT INTO pro_statistical_availability VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            (
                key,
                index,
                float(percentage),
                float(rain),
                float(cloud[index]),
                float(gas[index]),
                float(scintillation[index]),
                float(total[index]),
                result.disposition,
                "aggregate" if aggregate else "rain_only",
            )
            for index, (percentage, rain) in enumerate(
                zip(result.exceedance_percentage, result.rain_attenuation_db, strict=True)
            )
        ),
    )


def _insert_schedule(connection: sqlite3.Connection, key: str, result: ProScheduleResult) -> None:
    opportunity_by_id = {value.opportunity.opportunity_id: value for value in result.opportunities}
    connection.executemany(
        "INSERT INTO pro_schedule VALUES (?,?,?,?,?,?,?,?,?)",
        (
            (
                key,
                task.sequence,
                task.opportunity_id,
                task.start_s,
                task.end_s,
                opportunity_by_id[task.opportunity_id].estimated_data_bits,
                task.storage_after_bytes,
                task.cumulative_energy_wh,
                task.source_product_sha256,
            )
            for task in result.tasking.selected_tasks
        ),
    )
    rejections = dict(result.tasking.rejected_opportunity_reasons)
    rejections.update(
        {identifier: "bounded_policy_candidate_limit" for identifier in result.excluded_by_candidate_bound}
    )
    connection.executemany(
        "INSERT INTO pro_schedule_rejections VALUES (?,?,?)",
        ((key, identifier, reason) for identifier, reason in sorted(rejections.items())),
    )


def _insert_campaign(connection: sqlite3.Connection, key: str, result: ProCampaignResult) -> None:
    connection.executemany(
        "INSERT INTO pro_campaign_cases VALUES (?,?,?,?,?,?,?,?,?)",
        (
            (
                key,
                case.case_id,
                json.dumps(case.parameter_values, sort_keys=True, separators=(",", ":")),
                case.profile_sha256,
                case.link_semantic_sha256,
                case.minimum_margin_db,
                case.maximum_margin_db,
                case.available_duration_s,
                case.estimated_delivered_data_bits,
            )
            for case in result.cases
        ),
    )


def _insert_constellation(
    connection: sqlite3.Connection,
    key: str,
    result: ConstellationCoverageResult,
) -> None:
    connection.executemany(
        "INSERT INTO pro_constellation_samples VALUES (?,?,?,?,?,?,?)",
        (
            (
                key,
                index,
                float(result.times_s[index]),
                int(result.covered_cell_count[index]),
                float(result.instantaneous_covered_fraction[index]),
                int(result.maximum_multiplicity_by_sample[index]),
                int(result.active_asset_count_by_sample[index]),
            )
            for index in range(result.times_s.size)
        ),
    )
    connection.executemany(
        "INSERT INTO pro_constellation_cells VALUES (?,?,?,?,?,?)",
        (
            (
                key,
                int(result.cell_metrics.cell_index[index]),
                float(result.mean_multiplicity_per_cell[index]),
                int(result.max_multiplicity_per_cell[index]),
                float(result.cell_metrics.dwell_s[index]),
                int(result.cell_metrics.interval_count[index]),
            )
            for index in range(result.cell_metrics.cell_index.size)
        ),
    )


def _insert_campaign_convergence(
    connection: sqlite3.Connection,
    key: str,
    result: CampaignConvergenceResult,
) -> None:
    connection.execute(
        "INSERT INTO pro_campaign_convergence VALUES (?,?,?,?,?,?,?)",
        (
            key,
            result.campaign_semantic_sha256,
            int(result.passed),
            result.disposition,
            json.dumps(asdict(result.criteria), sort_keys=True, separators=(",", ":")),
            json.dumps(result.metrics, sort_keys=True, separators=(",", ":")),
            result.semantic_sha256,
        ),
    )


def _insert_sensitivity(
    connection: sqlite3.Connection,
    key: str,
    result: CoverageSensitivityResult,
) -> None:
    connection.execute(
        "INSERT INTO pro_coverage_sensitivity VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            key,
            result.comparison_kind,
            result.baseline_semantic_sha256,
            result.refined_semantic_sha256,
            result.baseline_order,
            result.refined_order,
            result.baseline_sample_count,
            result.refined_sample_count,
            int(result.passed),
            result.disposition,
            json.dumps(result.metrics, sort_keys=True, separators=(",", ":")),
            result.semantic_sha256,
        ),
    )


def _insert_design_trade(
    connection: sqlite3.Connection,
    key: str,
    result: ProDesignTradeResult,
) -> None:
    connection.executemany(
        "INSERT INTO pro_design_trade_cases VALUES (?,?,?,?,?,?)",
        (
            (
                key,
                case.rank,
                case.alternative_id,
                json.dumps(case.source_semantic_sha256, separators=(",", ":")),
                json.dumps(case.metrics, sort_keys=True, separators=(",", ":")),
                int(case.alternative_id == result.recommended_alternative_id),
            )
            for case in result.cases
        ),
    )


def _write_failed_bundle_markers(
    *,
    normalized_study_id: str,
    summary_path: Path,
    manifest_path: Path,
    database_path: Path,
    error: BaseException,
) -> None:
    """Leave inspectable failure evidence when bundle materialization aborts.

    The destination is intentionally retained.  A caller can inspect the
    partial database and failure manifest, while a subsequent retry must use a
    new output directory because the normal writer remains fail-on-existing.
    """

    failure = {
        "type": type(error).__name__,
        "message": str(error),
    }
    artifacts: dict[str, dict[str, str]] = {}
    if database_path.is_file() and not database_path.is_symlink():
        artifacts[database_path.name] = {
            "sha256": _sha256(database_path),
            "media_type": "application/vnd.sqlite3",
        }
    claims: list[str] = []
    non_claims = ["This evidence bundle is incomplete because materialization failed."]
    summary = {
        "manifest_version": PRO_EVIDENCE_MANIFEST_VERSION,
        "study_id": normalized_study_id,
        "status": "failed",
        "partial": True,
        "disposition": "failed_partial_bundle",
        "failure": failure,
        "artifacts": artifacts,
        "product_summaries": {},
        "claims": claims,
        "non_claims": non_claims,
        "review_tables": [],
    }
    _json(summary_path, summary)
    manifest = {
        "manifest_version": PRO_EVIDENCE_MANIFEST_VERSION,
        "study_id": normalized_study_id,
        "status": "failed",
        "partial": True,
        "disposition": "failed_partial_bundle",
        "failure": failure,
        "artifacts": {
            name: dict(metadata) for name, metadata in artifacts.items()
        },
        "source_products": {},
        "claims": claims,
        "non_claims": non_claims,
    }
    manifest["semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _json(manifest_path, manifest)


def write_pro_study_bundle(
    output_dir: str | Path,
    *,
    study_id: str,
    links: Mapping[str, ProLinkBudgetResult] | None = None,
    coverages: Mapping[str, ProCommunicationsCoverageResult] | None = None,
    statistical_availability: Mapping[
        str,
        StatisticalRainAvailabilityResult | StatisticalAggregateAvailabilityResult,
    ]
    | None = None,
    schedules: Mapping[str, ProScheduleResult] | None = None,
    campaigns: Mapping[str, ProCampaignResult] | None = None,
    campaign_convergence: Mapping[str, CampaignConvergenceResult] | None = None,
    constellations: Mapping[str, ConstellationCoverageResult] | None = None,
    coverage_sensitivity: Mapping[str, CoverageSensitivityResult] | None = None,
    design_trades: Mapping[str, ProDesignTradeResult] | None = None,
    study_provenance: Mapping[str, Any] | None = None,
    study_claims: tuple[str, ...] = (),
    study_non_claims: tuple[str, ...] = (),
) -> ProStudyArtifacts:
    """Write a fail-on-existing evidence directory with one read-only-friendly DB."""

    normalized_study_id = str(study_id or "").strip()
    if not normalized_study_id:
        raise ValueError("study_id must be a non-empty string.")
    links = links or {}
    coverages = coverages or {}
    statistical_availability = statistical_availability or {}
    schedules = schedules or {}
    campaigns = campaigns or {}
    campaign_convergence = campaign_convergence or {}
    constellations = constellations or {}
    coverage_sensitivity = coverage_sensitivity or {}
    design_trades = design_trades or {}
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Pro communications evidence directory already exists: {destination}")
    destination.mkdir(parents=True)
    summary_path = destination / "study_summary.json"
    database_path = destination / "review.sqlite"
    manifest_path = destination / "evidence_manifest.json"
    connection = sqlite3.connect(database_path)
    try:
        _create_schema(connection)
        for key, result in sorted(links.items()):
            _insert_link(connection, key, result)
        for key, result in sorted(coverages.items()):
            _insert_coverage(connection, key, result)
        for key, result in sorted(statistical_availability.items()):
            _insert_availability(connection, key, result)
        for key, result in sorted(schedules.items()):
            _insert_schedule(connection, key, result)
        for key, result in sorted(campaigns.items()):
            _insert_campaign(connection, key, result)
        for key, result in sorted(campaign_convergence.items()):
            _insert_campaign_convergence(connection, key, result)
        for key, result in sorted(constellations.items()):
            _insert_constellation(connection, key, result)
        for key, result in sorted(coverage_sensitivity.items()):
            _insert_sensitivity(connection, key, result)
        for key, result in sorted(design_trades.items()):
            _insert_design_trade(connection, key, result)
        connection.executemany(
            "INSERT INTO study_metadata VALUES (?,?)",
            (
                ("study_id", json.dumps(normalized_study_id)),
                ("manifest_version", json.dumps(PRO_EVIDENCE_MANIFEST_VERSION)),
                (
                    "study_provenance",
                    json.dumps(dict(study_provenance or {}), sort_keys=True, separators=(",", ":")),
                ),
            ),
        )
        connection.commit()
    except Exception as exc:
        connection.rollback()
        connection.close()
        try:
            _write_failed_bundle_markers(
                normalized_study_id=normalized_study_id,
                summary_path=summary_path,
                manifest_path=manifest_path,
                database_path=database_path,
                error=exc,
            )
        except Exception:
            # Preserve the original materialization error if a marker cannot
            # be written (for example, an exhausted or read-only destination).
            pass
        raise
    else:
        connection.close()
    product_summaries = {
        "links": {key: value.summary for key, value in sorted(links.items())},
        "coverages": {key: value.summary for key, value in sorted(coverages.items())},
        "statistical_availability": {key: value.summary for key, value in sorted(statistical_availability.items())},
        "schedules": {key: value.summary for key, value in sorted(schedules.items())},
        "campaigns": {key: value.summary for key, value in sorted(campaigns.items())},
        "campaign_convergence": {key: value.summary for key, value in sorted(campaign_convergence.items())},
        "constellations": {key: value.summary for key, value in sorted(constellations.items())},
        "coverage_sensitivity": {
            key: {
                **asdict(value),
                "claims": ["Finite-horizon source-bound cadence or resolution sensitivity."],
                "non_claims": ["Passing declared limits is not a convergence or operational-assurance claim."],
            }
            for key, value in sorted(coverage_sensitivity.items())
        },
        "design_trades": {key: value.summary for key, value in sorted(design_trades.items())},
    }
    claim_set = set(study_claims)
    non_claim_set = set(study_non_claims)
    for group in product_summaries.values():
        for product_summary in group.values():
            claim_set.update(product_summary.get("claims", []))
            non_claim_set.update(product_summary.get("non_claims", []))
    claims = sorted(claim_set)
    non_claims = sorted(non_claim_set)
    summary = {
        "manifest_version": PRO_EVIDENCE_MANIFEST_VERSION,
        "study_id": normalized_study_id,
        "status": "complete",
        "study_provenance": dict(study_provenance or {}),
        "product_summaries": product_summaries,
        "claims": claims,
        "non_claims": non_claims,
        "review_tables": [
            "pro_link_samples",
            "pro_link_windows",
            "pro_coverage_samples",
            "pro_coverage_cells",
            "pro_statistical_availability",
            "pro_schedule",
            "pro_schedule_rejections",
            "pro_campaign_cases",
            "pro_campaign_convergence",
            "pro_constellation_samples",
            "pro_constellation_cells",
            "pro_coverage_sensitivity",
            "pro_design_trade_cases",
        ],
    }
    _json(summary_path, summary)
    artifacts = {
        summary_path.name: {"sha256": _sha256(summary_path), "media_type": "application/json"},
        database_path.name: {"sha256": _sha256(database_path), "media_type": "application/vnd.sqlite3"},
    }
    manifest = {
        "manifest_version": PRO_EVIDENCE_MANIFEST_VERSION,
        "study_id": normalized_study_id,
        "status": "complete",
        "study_provenance": dict(study_provenance or {}),
        "artifacts": artifacts,
        "source_products": {
            "links": {key: value.semantic_sha256 for key, value in sorted(links.items())},
            "coverages": {key: value.semantic_sha256 for key, value in sorted(coverages.items())},
            "statistical_availability": {
                key: value.semantic_sha256 for key, value in sorted(statistical_availability.items())
            },
            "schedules": {key: value.semantic_sha256 for key, value in sorted(schedules.items())},
            "campaigns": {key: value.semantic_sha256 for key, value in sorted(campaigns.items())},
            "campaign_convergence": {key: value.semantic_sha256 for key, value in sorted(campaign_convergence.items())},
            "constellations": {key: value.interval_semantic_sha256 for key, value in sorted(constellations.items())},
            "coverage_sensitivity": {key: value.semantic_sha256 for key, value in sorted(coverage_sensitivity.items())},
            "design_trades": {key: value.semantic_sha256 for key, value in sorted(design_trades.items())},
        },
        "claims": claims,
        "non_claims": non_claims,
    }
    manifest["semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _json(manifest_path, manifest)
    return ProStudyArtifacts(
        output_dir=destination,
        manifest_json=manifest_path,
        summary_json=summary_path,
        review_sqlite=database_path,
    )


def register_bundle_artifacts(
    artifacts: ProStudyArtifacts,
    paths: Mapping[str, str | Path],
) -> None:
    """Content-bind additional generated files inside an existing bundle."""

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    for artifact_id, raw_path in sorted(paths.items()):
        path = Path(raw_path).expanduser().resolve()
        try:
            relative = path.relative_to(artifacts.output_dir)
        except ValueError as exc:
            raise ValueError("Registered bundle artifacts must be inside the evidence directory.") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest["artifacts"][str(relative)] = {
            "artifact_id": str(artifact_id),
            "sha256": _sha256(path),
            "media_type": "image/png" if path.suffix.lower() == ".png" else "application/octet-stream",
        }
    manifest.pop("semantic_sha256", None)
    manifest["semantic_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _json(artifacts.manifest_json, manifest)


__all__ = [
    "PRO_EVIDENCE_MANIFEST_VERSION",
    "ProStudyArtifacts",
    "register_bundle_artifacts",
    "write_pro_study_bundle",
]
