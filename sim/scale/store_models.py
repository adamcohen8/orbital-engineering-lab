from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCALE_STORE_SCHEMA_VERSION = "0.6"
SCALE_STORE_MIGRATABLE_SCHEMA_VERSIONS = {"0.4", "0.5"}

CORE_TABLES = (
    "store_metadata",
    "ingest_batches",
    "catalog_objects",
    "tle_history",
    "track_observations",
    "ground_station_measurement_stations",
    "ground_station_measurements",
    "ground_station_measurement_quality",
    "od_solutions",
    "maneuver_flags",
    "current_object_state",
    "propagation_products",
    "propagation_samples",
    "screening_windows",
    "conjunction_candidates",
    "candidate_refinements",
    "hpop_handoff_packets",
    "onp_refinement_cases",
    "onp_refinement_runs",
    "sensitivity_campaigns",
    "sensitivity_trials",
    "campaign_trial_results",
    "invalidation_queue",
)


@dataclass(frozen=True)
class StoreSummary:
    path: Path
    schema_version: str
    table_counts: dict[str, int]
    catalog_object_ids: tuple[str, ...]
    current_object_ids: tuple[str, ...]
    catalog_object_count: int
    current_object_state_count: int
    conjunction_candidate_count: int
    open_invalidation_count: int
    invalidation_by_reason: dict[str, int]


@dataclass(frozen=True)
class TLEIngestRecord:
    object_id: str
    primary_name: str
    line1: str
    line2: str
    norad_id: str
    cospar_id: str
    tle_epoch_jd_utc: float
    parsed_elements: dict[str, Any]
    digest: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TLEIngestResult:
    store_path: Path
    batch_id: str
    source_label: str
    source_digest: str
    records_seen: int
    objects_created: int
    tle_records_inserted: int
    od_solutions_inserted: int
    current_states_updated: int
    invalidations_inserted: int


@dataclass(frozen=True)
class TrackObservationRecord:
    object_id: str
    time_jd_utc: float
    frame: str
    position_km: tuple[float, float, float] | None = None
    velocity_km_s: tuple[float, float, float] | None = None
    covariance: dict[str, Any] | None = None
    quality_flags: dict[str, Any] | None = None
    digest: str = ""


@dataclass(frozen=True)
class TrackObservationIngestResult:
    store_path: Path
    batch_id: str
    source_label: str
    source_digest: str
    records_seen: int
    objects_created: int
    observations_inserted: int
    invalidations_inserted: int


@dataclass(frozen=True)
class GroundStationMeasurementRecord:
    object_id: str
    station_id: str
    time_jd_utc: float
    measurement_type: str
    components: tuple[str, ...]
    vector: tuple[float, ...]
    sigma: tuple[float, ...]
    station_metadata: dict[str, Any]
    quality_flags: dict[str, Any] | None = None
    truth_geometry: dict[str, Any] | None = None
    digest: str = ""


@dataclass(frozen=True)
class GroundStationMeasurementIngestResult:
    store_path: Path
    batch_id: str
    source_label: str
    source_digest: str
    records_seen: int
    objects_created: int
    stations_recorded: int
    measurements_inserted: int
    invalidations_inserted: int


@dataclass(frozen=True)
class GroundStationMeasurementQualityRecord:
    quality_id: str
    assessment_id: str
    measurement_id: str
    object_id: str
    station_id: str
    od_manifest_path: str
    od_report_json_path: str
    residual_kind: str
    time_s: float
    position_error_m: float
    normalized_position_error: float | None
    status: str
    flags: tuple[str, ...]
    metrics: dict[str, Any]
    thresholds: dict[str, Any]


@dataclass(frozen=True)
class CurrentTLEState:
    object_id: str
    od_solution_id: str
    input_digest: str
    line1: str
    line2: str
    tle_epoch_jd_utc: float
    mean_elements: dict[str, Any] = field(default_factory=dict)
    source_product_id: str = ""
    source_provenance: dict[str, Any] = field(default_factory=dict)


# OGP-first/current terminology; the historical TLE-specific name remains an import alias.
CurrentOGPState = CurrentTLEState


@dataclass(frozen=True)
class PropagationProductRecord:
    propagation_product_id: str
    object_id: str
    source_od_solution_id: str
    model: str
    backend: str
    precision: str
    native_frame: str
    output_frame: str
    frame_transform: str
    start_jd_utc: float
    end_jd_utc: float
    step_s: float
    sample_count: int
    input_digest: str
    samples: list[dict[str, Any]]
    status: str = "completed"
    error: str = ""
    ogp_regime: str = ""
    runtime_backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PropagationProductSummary:
    propagation_product_id: str
    object_id: str
    model: str
    backend: str
    output_frame: str
    start_jd_utc: float
    end_jd_utc: float
    step_s: float
    sample_count: int
    input_digest: str
    ogp_regime: str = ""
    runtime_backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PropagationSampleRecord:
    sample_index: int
    jd_utc: float
    t_s: float
    pos_x_km: float | None
    pos_y_km: float | None
    pos_z_km: float | None
    vel_x_km_s: float | None
    vel_y_km_s: float | None
    vel_z_km_s: float | None
    error: str


@dataclass(frozen=True)
class ScreeningWindowRecord:
    screening_window_id: str
    screening_run_id: str
    deputy_id: str
    chief_id: str
    window_start_jd_utc: float
    window_end_jd_utc: float
    method: str
    source_product_ids: tuple[str, str]
    input_digest: str
    min_sample_distance_km: float
    min_sample_time_jd_utc: float
    relative_speed_km_s: float
    pruning_reason: str
    status: str


@dataclass(frozen=True)
class ConjunctionCandidateRecord:
    candidate_id: str
    screening_run_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    tca_jd_utc: float
    miss_distance_km: float
    relative_speed_km_s: float
    ranking_score: float
    ranking_features: dict[str, Any]
    promotion_status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateForRefinement:
    candidate_id: str
    screening_run_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    window_start_jd_utc: float
    window_end_jd_utc: float
    source_product_ids: tuple[str, ...]
    sampled_tca_jd_utc: float
    sampled_miss_distance_km: float
    sampled_relative_speed_km_s: float
    ranking_features: dict[str, Any]


@dataclass(frozen=True)
class CandidateRefinementRecord:
    refinement_id: str
    candidate_id: str
    screening_run_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    method: str
    source_product_ids: tuple[str, ...]
    input_digest: str
    sampled_tca_jd_utc: float
    sampled_miss_distance_km: float
    sampled_relative_speed_km_s: float
    refinement_start_jd_utc: float
    refinement_end_jd_utc: float
    refinement_step_s: float
    refined_tca_jd_utc: float
    refined_miss_distance_km: float
    refined_relative_speed_km_s: float
    ranking_score: float
    ranking_features: dict[str, Any]
    status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HpopHandoffCase:
    candidate_id: str
    refinement_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    refined_tca_jd_utc: float
    refined_miss_distance_km: float
    refined_relative_speed_km_s: float
    ranking_score: float
    ranking_features: dict[str, Any]
    source_product_ids: tuple[str, ...]


@dataclass(frozen=True)
class HpopHandoffPacketRecord:
    packet_id: str
    candidate_id: str
    refinement_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    force_model_profile: str
    case_start_jd_utc: float
    case_end_jd_utc: float
    tca_jd_utc: float
    miss_distance_km: float
    relative_speed_km_s: float
    ranking_score: float
    output_dir: str
    packet_path: str
    packet_json: dict[str, Any]
    input_digest: str
    status: str = "planned"
    warnings: tuple[str, ...] = ()


# Canonical ONP names.  The database table and historical class objects retain
# their old names so existing stores, pickles, and imports remain readable.
OnpHandoffCase = HpopHandoffCase
OnpHandoffPacketRecord = HpopHandoffPacketRecord


@dataclass(frozen=True)
class OnpRefinementCaseRecord:
    case_id: str
    packet_id: str
    packet_input_digest: str
    candidate_id: str
    refinement_id: str
    screening_window_id: str
    deputy_id: str
    chief_id: str
    force_model_profile: str
    analysis_start_jd_utc: float
    analysis_end_jd_utc: float
    initial_state_frame: str
    frame_transform: str
    source_product_ids: tuple[str, ...]
    source_state_provenance: dict[str, Any]
    generated_config_path: str
    config_digest: str
    scenario_output_dir: str
    validation_status: str
    validation_report: dict[str, Any]
    materialization_signature: str
    input_digest: str
    status: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OnpRefinementRunRecord:
    execution_id: str
    batch_run_id: str
    case_id: str
    config_digest: str
    backend: str
    attempt: int
    status: str
    output_dir: str = ""
    review_store_path: str = ""
    run_summary: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    input_digest: str = ""
    error: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitivityCampaignRecord:
    campaign_id: str
    workflow_name: str
    campaign_type: str
    target: str
    uncertainty_interpretation: str
    seed: int
    requested_trial_count: int
    realized_trial_count: int
    sweep_mode: str
    parameters: tuple[dict[str, Any], ...]
    frame: dict[str, Any]
    trial_builder: dict[str, Any]
    outputs: dict[str, Any]
    output_dir: str
    input_digest: str
    status: str = "planned"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitivityTrialRecord:
    trial_id: str
    campaign_id: str
    trial_index: int
    parameters: dict[str, Any]
    target_object_ids: tuple[str, ...]
    frame: dict[str, Any]
    builder_status: str
    trial_product_refs: tuple[str, ...]
    input_digest: str
    status: str = "planned"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CampaignTrialResultRecord:
    result_id: str
    run_id: str
    campaign_id: str
    trial_id: str
    trial_index: int
    backend: str
    status: str
    builder_status: str
    metrics: dict[str, Any]
    product_refs: tuple[str, ...]
    input_digest: str
    error: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OdSolutionSummary:
    od_solution_id: str
    object_id: str
    method: str
    input_digest: str
    epoch_jd_utc: float
    mean_elements: dict[str, Any]
    status: str
    supersedes_product_id: str


@dataclass(frozen=True)
class OdSolutionRecord:
    od_solution_id: str
    object_id: str
    method: str
    source_input_ids: tuple[str, ...]
    input_digest: str
    epoch_jd_utc: float
    native_frame: str
    state: dict[str, Any]
    mean_elements: dict[str, Any]
    residual_summary: dict[str, Any]
    quality_flags: dict[str, Any]
    status: str


@dataclass(frozen=True)
class ManeuverFlagRecord:
    maneuver_flag_id: str
    object_id: str
    source_od_solution_id: str
    start_jd_utc: float
    end_jd_utc: float
    method: str
    confidence: float
    status: str
    summary: dict[str, Any]
