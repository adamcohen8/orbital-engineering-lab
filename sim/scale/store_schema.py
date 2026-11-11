from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sim.plotting.style import get_oel_version
from sim.scale.store_connection import (
    _utc_stamp,
)
from sim.scale.store_models import (
    SCALE_STORE_MIGRATABLE_SCHEMA_VERSIONS,
    SCALE_STORE_SCHEMA_VERSION,
)


def initialize_store(path: str | Path, *, description: str = "") -> dict[str, Any]:
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    created = not db_path.exists()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        existing_schema_version = _existing_schema_version(conn)
        if existing_schema_version not in {None, SCALE_STORE_SCHEMA_VERSION, *SCALE_STORE_MIGRATABLE_SCHEMA_VERSIONS}:
            raise ValueError(
                "Unsupported OEL Scale store schema version "
                f"{existing_schema_version!r}; expected {SCALE_STORE_SCHEMA_VERSION!r}."
            )
        _create_schema(conn)
        _migrate_schema(conn, from_version=existing_schema_version)
        _upsert_metadata(conn, description=description)
        conn.commit()
    return {"path": str(db_path), "created": created, "schema_version": SCALE_STORE_SCHEMA_VERSION}
def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS store_metadata (
            store_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            oel_version TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS ingest_batches (
            batch_id TEXT PRIMARY KEY,
            source_label TEXT,
            source_type TEXT,
            source_digest TEXT,
            normalized_by TEXT,
            created_utc TEXT NOT NULL,
            metadata_json TEXT,
            warnings_json TEXT
        );

        CREATE TABLE IF NOT EXISTS catalog_objects (
            object_id TEXT PRIMARY KEY,
            primary_name TEXT,
            object_class TEXT,
            norad_id TEXT,
            cospar_id TEXT,
            aliases_json TEXT,
            tags_json TEXT,
            priority REAL,
            metadata_json TEXT,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_catalog_objects_norad ON catalog_objects(norad_id);

        CREATE TABLE IF NOT EXISTS tle_history (
            tle_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            batch_id TEXT,
            line1 TEXT NOT NULL,
            line2 TEXT NOT NULL,
            tle_epoch_jd_utc REAL,
            checksum_valid INTEGER,
            parsed_elements_json TEXT,
            source_label TEXT,
            digest TEXT,
            created_utc TEXT NOT NULL,
            warnings_json TEXT,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tle_history_digest ON tle_history(digest);
        CREATE INDEX IF NOT EXISTS idx_tle_history_object_epoch ON tle_history(object_id, tle_epoch_jd_utc);

        CREATE TABLE IF NOT EXISTS track_observations (
            observation_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            batch_id TEXT,
            time_jd_utc REAL NOT NULL,
            frame TEXT,
            position_x_km REAL,
            position_y_km REAL,
            position_z_km REAL,
            velocity_x_km_s REAL,
            velocity_y_km_s REAL,
            velocity_z_km_s REAL,
            covariance_json TEXT,
            quality_flags_json TEXT,
            source_label TEXT,
            digest TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_track_observations_object_time ON track_observations(object_id, time_jd_utc);

        CREATE TABLE IF NOT EXISTS ground_station_measurement_stations (
            batch_id TEXT NOT NULL,
            station_id TEXT NOT NULL,
            station_metadata_json TEXT,
            source_label TEXT,
            created_utc TEXT NOT NULL,
            PRIMARY KEY (batch_id, station_id),
            FOREIGN KEY(batch_id) REFERENCES ingest_batches(batch_id)
        );

        CREATE TABLE IF NOT EXISTS ground_station_measurements (
            measurement_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            station_id TEXT NOT NULL,
            batch_id TEXT,
            time_jd_utc REAL NOT NULL,
            measurement_type TEXT,
            components_json TEXT,
            vector_json TEXT,
            sigma_json TEXT,
            quality_flags_json TEXT,
            truth_geometry_json TEXT,
            source_label TEXT,
            digest TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id),
            FOREIGN KEY(batch_id, station_id) REFERENCES ground_station_measurement_stations(batch_id, station_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ground_station_measurements_digest
        ON ground_station_measurements(digest);
        CREATE INDEX IF NOT EXISTS idx_ground_station_measurements_object_time
        ON ground_station_measurements(object_id, time_jd_utc);
        CREATE INDEX IF NOT EXISTS idx_ground_station_measurements_station_time
        ON ground_station_measurements(station_id, time_jd_utc);

        CREATE TABLE IF NOT EXISTS ground_station_measurement_quality (
            quality_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL,
            measurement_id TEXT NOT NULL,
            object_id TEXT NOT NULL,
            station_id TEXT NOT NULL,
            od_manifest_path TEXT,
            od_report_json_path TEXT,
            residual_kind TEXT,
            time_s REAL,
            position_error_m REAL,
            normalized_position_error REAL,
            status TEXT,
            flags_json TEXT,
            metrics_json TEXT,
            thresholds_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(measurement_id) REFERENCES ground_station_measurements(measurement_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ground_station_measurement_quality_assessment
        ON ground_station_measurement_quality(assessment_id, measurement_id, residual_kind);
        CREATE INDEX IF NOT EXISTS idx_ground_station_measurement_quality_status
        ON ground_station_measurement_quality(object_id, station_id, status);

        CREATE TABLE IF NOT EXISTS od_solutions (
            od_solution_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            method TEXT NOT NULL,
            source_input_ids_json TEXT,
            input_digest TEXT,
            epoch_jd_utc REAL,
            native_frame TEXT,
            state_json TEXT,
            mean_elements_json TEXT,
            residual_summary_json TEXT,
            quality_flags_json TEXT,
            status TEXT,
            created_utc TEXT NOT NULL,
            supersedes_product_id TEXT,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_od_solutions_object_epoch ON od_solutions(object_id, epoch_jd_utc);

        CREATE TABLE IF NOT EXISTS maneuver_flags (
            maneuver_flag_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            source_od_solution_id TEXT,
            start_jd_utc REAL,
            end_jd_utc REAL,
            method TEXT,
            confidence REAL,
            status TEXT,
            summary_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_maneuver_flags_object_window ON maneuver_flags(object_id, start_jd_utc, end_jd_utc);

        CREATE TABLE IF NOT EXISTS current_object_state (
            object_id TEXT PRIMARY KEY,
            current_policy TEXT NOT NULL,
            od_solution_id TEXT,
            selected_utc TEXT NOT NULL,
            selection_reason TEXT,
            warnings_json TEXT,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );

        CREATE TABLE IF NOT EXISTS propagation_products (
            propagation_product_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            source_od_solution_id TEXT,
            model TEXT,
            backend TEXT,
            ogp_regime TEXT,
            runtime_backend TEXT,
            precision TEXT,
            native_frame TEXT,
            output_frame TEXT,
            frame_transform TEXT,
            start_jd_utc REAL,
            end_jd_utc REAL,
            step_s REAL,
            sample_count INTEGER,
            input_digest TEXT,
            storage_kind TEXT,
            storage_uri TEXT,
            storage_digest TEXT,
            shape_json TEXT,
            metadata_json TEXT,
            status TEXT,
            error TEXT,
            created_utc TEXT NOT NULL,
            supersedes_product_id TEXT,
            FOREIGN KEY(object_id) REFERENCES catalog_objects(object_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_propagation_products_input_digest
        ON propagation_products(input_digest);
        CREATE INDEX IF NOT EXISTS idx_propagation_products_object_window
        ON propagation_products(object_id, start_jd_utc, end_jd_utc);

        CREATE TABLE IF NOT EXISTS propagation_samples (
            propagation_product_id TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            jd_utc REAL NOT NULL,
            t_s REAL NOT NULL,
            pos_x_km REAL,
            pos_y_km REAL,
            pos_z_km REAL,
            vel_x_km_s REAL,
            vel_y_km_s REAL,
            vel_z_km_s REAL,
            error TEXT,
            PRIMARY KEY (propagation_product_id, sample_index),
            FOREIGN KEY(propagation_product_id) REFERENCES propagation_products(propagation_product_id)
        );
        CREATE INDEX IF NOT EXISTS idx_propagation_samples_product_time
        ON propagation_samples(propagation_product_id, jd_utc);

        CREATE TABLE IF NOT EXISTS screening_windows (
            screening_window_id TEXT PRIMARY KEY,
            screening_run_id TEXT,
            deputy_id TEXT NOT NULL,
            chief_id TEXT NOT NULL,
            window_start_jd_utc REAL,
            window_end_jd_utc REAL,
            method TEXT,
            source_product_ids_json TEXT,
            input_digest TEXT,
            min_sample_distance_km REAL,
            min_sample_time_jd_utc REAL,
            relative_speed_km_s REAL,
            pruning_reason TEXT,
            status TEXT,
            created_utc TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_screening_windows_input_digest
        ON screening_windows(input_digest);
        CREATE INDEX IF NOT EXISTS idx_screening_windows_pair_window
        ON screening_windows(deputy_id, chief_id, window_start_jd_utc, window_end_jd_utc);

        CREATE TABLE IF NOT EXISTS conjunction_candidates (
            candidate_id TEXT PRIMARY KEY,
            screening_run_id TEXT,
            screening_window_id TEXT,
            deputy_id TEXT NOT NULL,
            chief_id TEXT NOT NULL,
            tca_jd_utc REAL,
            miss_distance_km REAL,
            relative_speed_km_s REAL,
            ranking_score REAL,
            ranking_features_json TEXT,
            promotion_status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            supersedes_candidate_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_conjunction_candidates_pair_tca
        ON conjunction_candidates(deputy_id, chief_id, tca_jd_utc);

        CREATE TABLE IF NOT EXISTS candidate_refinements (
            refinement_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            screening_run_id TEXT,
            screening_window_id TEXT,
            deputy_id TEXT NOT NULL,
            chief_id TEXT NOT NULL,
            method TEXT,
            source_product_ids_json TEXT,
            input_digest TEXT,
            sampled_tca_jd_utc REAL,
            sampled_miss_distance_km REAL,
            sampled_relative_speed_km_s REAL,
            refinement_start_jd_utc REAL,
            refinement_end_jd_utc REAL,
            refinement_step_s REAL,
            refined_tca_jd_utc REAL,
            refined_miss_distance_km REAL,
            refined_relative_speed_km_s REAL,
            ranking_score REAL,
            ranking_features_json TEXT,
            status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES conjunction_candidates(candidate_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_refinements_input_digest
        ON candidate_refinements(input_digest);
        CREATE INDEX IF NOT EXISTS idx_candidate_refinements_candidate
        ON candidate_refinements(candidate_id, status);

        CREATE TABLE IF NOT EXISTS hpop_handoff_packets (
            packet_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            refinement_id TEXT,
            screening_window_id TEXT,
            deputy_id TEXT NOT NULL,
            chief_id TEXT NOT NULL,
            force_model_profile TEXT,
            case_start_jd_utc REAL,
            case_end_jd_utc REAL,
            tca_jd_utc REAL,
            miss_distance_km REAL,
            relative_speed_km_s REAL,
            ranking_score REAL,
            output_dir TEXT,
            packet_path TEXT,
            packet_json TEXT,
            input_digest TEXT,
            status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES conjunction_candidates(candidate_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_hpop_handoff_packets_input_digest
        ON hpop_handoff_packets(input_digest);
        CREATE INDEX IF NOT EXISTS idx_hpop_handoff_packets_candidate
        ON hpop_handoff_packets(candidate_id, status);

        CREATE TABLE IF NOT EXISTS onp_refinement_cases (
            case_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
            packet_input_digest TEXT,
            candidate_id TEXT NOT NULL,
            refinement_id TEXT,
            screening_window_id TEXT,
            deputy_id TEXT NOT NULL,
            chief_id TEXT NOT NULL,
            force_model_profile TEXT,
            analysis_start_jd_utc REAL,
            analysis_end_jd_utc REAL,
            initial_state_frame TEXT,
            frame_transform TEXT,
            source_product_ids_json TEXT,
            source_state_provenance_json TEXT,
            generated_config_path TEXT,
            config_digest TEXT,
            scenario_output_dir TEXT,
            validation_status TEXT,
            validation_report_json TEXT,
            materialization_signature TEXT,
            input_digest TEXT,
            status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            updated_utc TEXT NOT NULL,
            FOREIGN KEY(packet_id) REFERENCES hpop_handoff_packets(packet_id),
            FOREIGN KEY(candidate_id) REFERENCES conjunction_candidates(candidate_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_onp_refinement_cases_input_digest
        ON onp_refinement_cases(input_digest);
        CREATE INDEX IF NOT EXISTS idx_onp_refinement_cases_packet
        ON onp_refinement_cases(packet_id, status);
        CREATE INDEX IF NOT EXISTS idx_onp_refinement_cases_candidate
        ON onp_refinement_cases(candidate_id, status);

        CREATE TABLE IF NOT EXISTS onp_refinement_runs (
            execution_id TEXT PRIMARY KEY,
            batch_run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            config_digest TEXT NOT NULL,
            backend TEXT,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            output_dir TEXT,
            review_store_path TEXT,
            run_summary_json TEXT,
            comparison_json TEXT,
            input_digest TEXT,
            error TEXT,
            warnings_json TEXT,
            claimed_utc TEXT NOT NULL,
            started_utc TEXT,
            completed_utc TEXT,
            updated_utc TEXT NOT NULL,
            FOREIGN KEY(case_id) REFERENCES onp_refinement_cases(case_id),
            UNIQUE(case_id, config_digest, attempt)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_onp_refinement_runs_input_digest
        ON onp_refinement_runs(input_digest);
        CREATE INDEX IF NOT EXISTS idx_onp_refinement_runs_case
        ON onp_refinement_runs(case_id, status, attempt);
        CREATE INDEX IF NOT EXISTS idx_onp_refinement_runs_batch
        ON onp_refinement_runs(batch_run_id, status);

        CREATE TABLE IF NOT EXISTS sensitivity_campaigns (
            campaign_id TEXT PRIMARY KEY,
            workflow_name TEXT,
            campaign_type TEXT,
            target TEXT,
            uncertainty_interpretation TEXT,
            seed INTEGER,
            requested_trial_count INTEGER,
            realized_trial_count INTEGER,
            sweep_mode TEXT,
            parameters_json TEXT,
            frame_json TEXT,
            trial_builder_json TEXT,
            outputs_json TEXT,
            output_dir TEXT,
            input_digest TEXT,
            status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sensitivity_campaigns_input_digest
        ON sensitivity_campaigns(input_digest);

        CREATE TABLE IF NOT EXISTS sensitivity_trials (
            trial_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL,
            trial_index INTEGER NOT NULL,
            parameters_json TEXT,
            target_object_ids_json TEXT,
            frame_json TEXT,
            builder_status TEXT,
            trial_product_refs_json TEXT,
            input_digest TEXT,
            status TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES sensitivity_campaigns(campaign_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sensitivity_trials_input_digest
        ON sensitivity_trials(input_digest);
        CREATE INDEX IF NOT EXISTS idx_sensitivity_trials_campaign
        ON sensitivity_trials(campaign_id, trial_index);

        CREATE TABLE IF NOT EXISTS campaign_trial_results (
            result_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            trial_id TEXT NOT NULL,
            trial_index INTEGER NOT NULL,
            backend TEXT,
            status TEXT,
            builder_status TEXT,
            metrics_json TEXT,
            product_refs_json TEXT,
            input_digest TEXT,
            error TEXT,
            warnings_json TEXT,
            created_utc TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES sensitivity_campaigns(campaign_id),
            FOREIGN KEY(trial_id) REFERENCES sensitivity_trials(trial_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_trial_results_input_digest
        ON campaign_trial_results(input_digest);
        CREATE INDEX IF NOT EXISTS idx_campaign_trial_results_run
        ON campaign_trial_results(run_id, trial_index);
        CREATE INDEX IF NOT EXISTS idx_campaign_trial_results_campaign
        ON campaign_trial_results(campaign_id, trial_index);

        CREATE TABLE IF NOT EXISTS invalidation_queue (
            invalidation_id TEXT PRIMARY KEY,
            object_id TEXT,
            deputy_id TEXT,
            chief_id TEXT,
            time_start_jd_utc REAL,
            time_end_jd_utc REAL,
            stale_product_id TEXT,
            stale_product_type TEXT,
            reason TEXT NOT NULL,
            priority REAL,
            status TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            claimed_utc TEXT,
            completed_utc TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_invalidation_status_priority
        ON invalidation_queue(status, priority);
        """
    )


def _existing_schema_version(conn: sqlite3.Connection) -> str | None:
    table = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'store_metadata'").fetchone()
    if table is None:
        other_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        if other_tables is not None:
            raise ValueError("Existing SQLite file is not a versioned OEL Scale store.")
        return None
    row = conn.execute("SELECT schema_version FROM store_metadata LIMIT 1").fetchone()
    if row is None or not str(row[0] or "").strip():
        raise ValueError("Existing OEL Scale store is missing its schema version.")
    return str(row[0]).strip()


def _migrate_schema(conn: sqlite3.Connection, *, from_version: str | None) -> None:
    if from_version not in {None, "0.4", "0.5", SCALE_STORE_SCHEMA_VERSION}:
        raise ValueError(f"No OEL Scale store migration is available from schema {from_version!r}.")
    _ensure_columns(
        conn,
        table="propagation_products",
        columns={
            "ogp_regime": "TEXT",
            "runtime_backend": "TEXT",
            "metadata_json": "TEXT",
        },
    )
    _ensure_columns(
        conn,
        table="onp_refinement_cases",
        columns={"materialization_signature": "TEXT"},
    )


def _ensure_columns(conn: sqlite3.Connection, *, table: str, columns: dict[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _upsert_metadata(conn: sqlite3.Connection, *, description: str) -> None:
    now = _utc_stamp()
    existing = conn.execute("SELECT store_id FROM store_metadata LIMIT 1").fetchone()
    if existing:
        conn.execute(
            """
            UPDATE store_metadata
            SET schema_version = ?, updated_utc = ?, oel_version = ?, description = COALESCE(NULLIF(?, ''), description)
            WHERE store_id = ?
            """,
            (SCALE_STORE_SCHEMA_VERSION, now, get_oel_version(), description, existing[0]),
        )
        return
    conn.execute(
        """
        INSERT INTO store_metadata (
            store_id, schema_version, created_utc, updated_utc, oel_version, description
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), SCALE_STORE_SCHEMA_VERSION, now, now, get_oel_version(), description),
    )
