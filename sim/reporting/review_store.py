from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import iter_object_sections, relative_reference_for_object
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.plotting.style import get_oel_version
from sim.review.generated_artifacts import clear_generated_review_artifacts
from sim.utils.frames import eci_relative_to_ric_rect

REVIEW_SCHEMA_VERSION = "0.13"
REVIEW_SCHEMA_COMPATIBILITY_POLICY = "pre_1_0_additive"
REVIEW_SCHEMA_STABLE_TABLES = (
    "run_metadata",
    "objects",
    "time_samples",
    "object_state",
    "impulsive_maneuvers",
    "object_state_covariance",
    "relative_state",
    "thrust",
    "controller_decisions",
    "mission_modes",
    "mission_transitions",
    "command_gates",
    "metrics",
    "artifacts",
    "fsw_invocations",
    "fsw_input_events",
    "actuator_commands",
    "actuator_command_receipts",
    "actuator_realization",
)


def write_single_run_review_store(
    *,
    payload: dict[str, Any],
    context: Any,
    artifacts: dict[str, Any],
    config_json: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, str]:
    """Write a SQLite review store for a completed single-run output folder."""

    cfg = context.cfg
    if cfg.simulator.central_body != "earth":
        from sim.reporting.planetary_review import write_review
        return write_review(payload=payload,context=context,artifacts=artifacts,
                            config_json=config_json,config_sha256=config_sha256)
    review_cfg = cfg.outputs.review
    if not bool(review_cfg.enabled):
        return {}

    outdir = Path(context.outdir)
    review_dir = outdir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    db_path = review_dir / "run.sqlite"
    tmp_path = review_dir / "run.sqlite.tmp"
    schema_path = review_dir / "schema.json"
    if tmp_path.exists():
        tmp_path.unlink()

    summary = dict(payload.get("summary", {}) or {})
    t_s = np.asarray(context.t_s, dtype=float).reshape(-1)
    truth_hist = {str(k): _as_2d_float(v) for k, v in dict(context.truth_hist or {}).items()}
    thrust_hist = {str(k): _as_2d_float(v) for k, v in dict(context.thrust_hist or {}).items()}
    generated_utc = str(summary.get("generated_utc") or _utc_stamp())
    include_standard_detail = str(review_cfg.detail or "standard") != "compact"
    include_debug_detail = str(review_cfg.detail or "standard") == "full"

    try:
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            _create_schema(conn)
            _insert_run_metadata(
                conn,
                cfg=cfg,
                summary=summary,
                outdir=outdir,
                generated_utc=generated_utc,
                config_json=config_json,
                config_sha256=config_sha256,
            )
            _insert_objects(conn, cfg=cfg, summary=summary)
            _insert_impulsive_maneuvers(conn, payload=payload)
            _insert_frame_provenance(conn, payload=payload)
            _insert_object_initialization(conn, payload=payload)
            _insert_object_propagation(conn, payload=payload)
            _insert_object_state_frame(conn, payload=payload)
            if "additional_gravity" in payload:
                from sim.reporting.additional_gravity import insert_review
                insert_review(conn, payload, include_samples=include_standard_detail)
            if include_standard_detail:
                _insert_time_samples(conn, t_s=t_s)
                _insert_object_state(conn, t_s=t_s, truth_hist=truth_hist)
                _insert_object_orbital_elements(
                    conn,
                    t_s=t_s,
                    truth_hist=truth_hist,
                    object_state_frames=dict(payload.get("object_state_frames", {}) or {}),
                )
                _insert_attitude_error(conn, t_s=t_s, truth_hist=truth_hist, payload=payload)
                _insert_relative_state(
                    conn,
                    t_s=t_s,
                    truth_hist=truth_hist,
                    summary=summary,
                    cfg=cfg,
                    object_state_frames=dict(payload.get("object_state_frames", {}) or {}),
                )
                _insert_thrust(conn, t_s=t_s, thrust_hist=thrust_hist)
                _insert_command_decisions(
                    conn,
                    payload=payload,
                    include_debug_detail=include_debug_detail,
                )
                _insert_flight_software_evidence(
                    conn,
                    payload=payload,
                    t_s=t_s,
                    truth_hist=truth_hist,
                    include_debug_detail=include_debug_detail,
                )
                _insert_game_evidence(conn, payload=payload)
                _insert_ground_access(conn, t_s=t_s, payload=payload)
                _insert_orbital_analysis(conn, payload=payload)
            from sim.spacecraft_resources.artifacts import insert_resource_review

            insert_resource_review(conn, payload.get("spacecraft_resources", {}))
            from sim.ground_segment.artifacts import insert_ground_review
            insert_ground_review(conn, payload.get("ground_segment", []))
            _insert_events(conn, t_s=t_s, summary=summary, thrust_hist=thrust_hist,
                           impulsive_maneuvers=payload.get("impulsive_maneuvers", ()),
                           collision_events=payload.get("collision_events", ()))
            _insert_mission_recovery(conn, summary=summary)
            _insert_metrics(conn, summary=summary)
            _insert_artifacts(conn, artifacts=artifacts, outdir=outdir, generated_utc=generated_utc)
            conn.commit()
        finally:
            conn.close()
        clear_generated_review_artifacts(review_dir)
        tmp_path.replace(db_path)
        _write_schema_json(schema_path, generated_utc=generated_utc, db_path=db_path)
        return {
            "sqlite": str(db_path),
            "schema_json": str(schema_path),
        }
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def refresh_review_schema(db_path: str | Path) -> Path:
    """Regenerate the schema sidecar after an approved transactional extension."""

    database = Path(db_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Review database not found: {database}")
    schema_path = database.parent / "schema.json"
    _write_schema_json(schema_path, generated_utc=_utc_stamp(), db_path=database)
    return schema_path


def _create_schema(conn: sqlite3.Connection) -> None:
    schema_sql = (
        """
        BEGIN;
        CREATE TABLE run_metadata (
            run_id TEXT PRIMARY KEY,
            scenario_name TEXT,
            scenario_description TEXT,
            oel_version TEXT,
            review_schema_version TEXT,
            generated_utc TEXT,
            duration_s REAL,
            dt_s REAL,
            samples INTEGER,
            output_dir TEXT,
            config_path TEXT,
            config_sha256 TEXT,
            config_json TEXT,
            summary_json_path TEXT,
            run_log_json_path TEXT
        );

        CREATE TABLE objects (
            object_id TEXT PRIMARY KEY,
            object_type TEXT,
            enabled INTEGER,
            mass_initial_kg REAL,
            role TEXT,
            runtime_profile TEXT,
            flight_software_stack TEXT
        );

        CREATE TABLE frame_provenance (
            scope TEXT PRIMARY KEY,
            model TEXT,
            legacy_frame_model TEXT,
            time_scale_model TEXT,
            eop_path TEXT,
            tt_minus_utc_s REAL,
            dut1_s REAL,
            xp_arcsec REAL,
            yp_arcsec REAL,
            dat_s REAL,
            ddpsi_rad REAL,
            ddeps_rad REAL,
            polar_motion_applied INTEGER,
            nutation_corrections_applied INTEGER,
            sample_t_s REAL
        );

        CREATE TABLE object_propagation (
            object_id TEXT PRIMARY KEY,
            propagation_method TEXT,
            propagator_family TEXT,
            propagator_name TEXT,
            general_model TEXT,
            native_frame TEXT,
            output_frame TEXT,
            state_history_frame TEXT,
            frame_transform TEXT,
            tle_epoch_jd_utc REAL,
            tle_age_start_days REAL,
            tle_age_end_days REAL,
            ogp_regime TEXT,
            orbital_period_min REAL,
            max_tle_age_days_warning REAL,
            tle_age_warning INTEGER
        );

        CREATE TABLE object_initialization (
            object_id TEXT PRIMARY KEY,
            source TEXT,
            initialization_model TEXT,
            initialization_propagator_family TEXT,
            initialization_propagator_name TEXT,
            handoff_propagation_method TEXT,
            native_frame TEXT,
            output_frame TEXT,
            frame_transform TEXT,
            tle_epoch_jd_utc REAL,
            initial_jd_utc REAL,
            tle_age_initialization_days REAL,
            propagate_to_initial_epoch INTEGER,
            simulation_duration_s REAL,
            note TEXT
        );

        CREATE TABLE object_state_frame (
            object_id TEXT PRIMARY KEY,
            state_frame TEXT NOT NULL
        );

        CREATE TABLE time_samples (
            sample_index INTEGER PRIMARY KEY,
            time_s REAL NOT NULL
        );

        CREATE TABLE object_state (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            pos_x_eci_km REAL,
            pos_y_eci_km REAL,
            pos_z_eci_km REAL,
            vel_x_eci_km_s REAL,
            vel_y_eci_km_s REAL,
            vel_z_eci_km_s REAL,
            quat_w REAL,
            quat_x REAL,
            quat_y REAL,
            quat_z REAL,
            omega_x_rad_s REAL,
            omega_y_rad_s REAL,
            omega_z_rad_s REAL,
            mass_kg REAL
        );
        CREATE INDEX idx_object_state_object_time ON object_state(object_id, time_s);

        CREATE TABLE impulsive_maneuvers (
            maneuver_id TEXT PRIMARY KEY,
            object_id TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            time_s REAL NOT NULL,
            frame TEXT NOT NULL,
            dv_x_eci_m_s REAL NOT NULL,
            dv_y_eci_m_s REAL NOT NULL,
            dv_z_eci_m_s REAL NOT NULL,
            pre_pos_x_eci_km REAL NOT NULL,
            pre_pos_y_eci_km REAL NOT NULL,
            pre_pos_z_eci_km REAL NOT NULL,
            pre_vel_x_eci_km_s REAL NOT NULL,
            pre_vel_y_eci_km_s REAL NOT NULL,
            pre_vel_z_eci_km_s REAL NOT NULL,
            post_pos_x_eci_km REAL NOT NULL,
            post_pos_y_eci_km REAL NOT NULL,
            post_pos_z_eci_km REAL NOT NULL,
            post_vel_x_eci_km_s REAL NOT NULL,
            post_vel_y_eci_km_s REAL NOT NULL,
            post_vel_z_eci_km_s REAL NOT NULL
        );
        CREATE INDEX idx_impulsive_maneuvers_object_time ON impulsive_maneuvers(object_id, time_s);

        CREATE TABLE object_state_covariance (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            frame TEXT,
            component_order_json TEXT,
            units_json TEXT,
            covariance_json TEXT,
            mathematically_valid INTEGER,
            calibrated INTEGER,
            calibration_scope TEXT,
            source TEXT
        );
        CREATE INDEX idx_object_state_covariance_object_time
        ON object_state_covariance(object_id, time_s);

        CREATE TABLE relative_state (
            sample_index INTEGER,
            time_s REAL,
            deputy_id TEXT,
            chief_id TEXT,
            r_radial_km REAL,
            i_intrack_km REAL,
            c_crosstrack_km REAL,
            v_radial_km_s REAL,
            v_intrack_km_s REAL,
            v_crosstrack_km_s REAL,
            range_km REAL,
            range_rate_km_s REAL
        );
        CREATE INDEX idx_relative_state_pair_time ON relative_state(deputy_id, chief_id, time_s);

        CREATE TABLE thrust (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            accel_x_eci_km_s2 REAL,
            accel_y_eci_km_s2 REAL,
            accel_z_eci_km_s2 REAL,
            accel_norm_km_s2 REAL,
            burn_active INTEGER
        );
        CREATE INDEX idx_thrust_object_time ON thrust(object_id, time_s);

        CREATE TABLE controller_decisions (
            decision_index INTEGER,
            sample_index INTEGER,
            time_s REAL,
            interval_end_time_s REAL,
            dt_s REAL,
            object_id TEXT,
            orbit_controller TEXT,
            attitude_controller TEXT,
            mission_strategy TEXT,
            mission_execution TEXT,
            requested_accel_norm_km_s2 REAL,
            applied_accel_norm_km_s2 REAL,
            requested_torque_norm_nm REAL,
            applied_torque_norm_nm REAL,
            burn_requested INTEGER,
            burn_applied INTEGER,
            saturated INTEGER,
            deadline_missed INTEGER,
            field_sources_json TEXT,
            collisions_json TEXT,
            detail_json TEXT
        );
        CREATE INDEX idx_controller_decisions_object_time
        ON controller_decisions(object_id, time_s);

        CREATE TABLE mission_modes (
            decision_index INTEGER,
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            mission_strategy TEXT,
            mission_execution TEXT,
            mission_phase TEXT,
            executive_mode TEXT,
            mission_mode_json TEXT
        );
        CREATE INDEX idx_mission_modes_object_time ON mission_modes(object_id, time_s);

        CREATE TABLE mission_transitions (
            decision_index INTEGER,
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            from_mode TEXT,
            to_mode TEXT,
            trigger TEXT,
            reason TEXT,
            detail_json TEXT
        );
        CREATE INDEX idx_mission_transitions_object_time ON mission_transitions(object_id, time_s);

        CREATE TABLE command_gates (
            decision_index INTEGER,
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            burn_requested INTEGER,
            burn_applied INTEGER,
            alignment_error_rad REAL,
            alignment_ok INTEGER,
            fuel_depleted INTEGER,
            actuator_limited INTEGER,
            deadline_missed INTEGER,
            gate_reason TEXT,
            detail_json TEXT
        );
        CREATE INDEX idx_command_gates_object_time ON command_gates(object_id, time_s);

        CREATE TABLE fsw_invocations (
            object_id TEXT,
            invocation_id INTEGER,
            invocation_time_ns INTEGER,
            stack_id TEXT,
            stack_version TEXT,
            profile_id TEXT,
            input_count INTEGER,
            command_count INTEGER,
            telemetry_count INTEGER,
            detail_json TEXT,
            PRIMARY KEY (object_id, invocation_id)
        );

        CREATE TABLE fsw_input_events (
            object_id TEXT,
            packet_source_id TEXT,
            packet_boot_id TEXT,
            packet_sequence INTEGER,
            invocation_id INTEGER,
            kind TEXT,
            source_time_ns INTEGER,
            delivery_time_ns INTEGER,
            schema TEXT,
            detail_json TEXT,
            PRIMARY KEY (object_id, packet_source_id, packet_boot_id, packet_sequence)
        );
        CREATE INDEX idx_fsw_input_events_invocation
        ON fsw_input_events(object_id, invocation_id);

        CREATE TABLE fsw_load_events (
            object_id TEXT,
            invocation_id INTEGER,
            load_id TEXT,
            revision INTEGER,
            disposition TEXT,
            detail_json TEXT
        );

        CREATE TABLE fsw_objectives (
            object_id TEXT,
            invocation_id INTEGER,
            objective_id TEXT,
            state TEXT,
            detail_json TEXT
        );

        CREATE TABLE fsw_task_timing (
            object_id TEXT,
            invocation_id INTEGER,
            task_id TEXT,
            release_time_ns INTEGER,
            modeled_execution_duration_ns INTEGER,
            execution_budget_ns INTEGER,
            deadline_missed INTEGER,
            detail_json TEXT
        );

        CREATE TABLE actuator_commands (
            object_id TEXT,
            invocation_id INTEGER,
            command_source_id TEXT,
            command_boot_id TEXT,
            command_sequence INTEGER,
            actuator_id TEXT,
            issued_time_ns INTEGER,
            not_before_ns INTEGER,
            expires_at_ns INTEGER,
            command_schema TEXT,
            detail_json TEXT,
            PRIMARY KEY (object_id, command_source_id, command_boot_id, command_sequence)
        );
        CREATE INDEX idx_actuator_commands_invocation
        ON actuator_commands(object_id, invocation_id);

        CREATE TABLE actuator_command_receipts (
            object_id TEXT,
            command_source_id TEXT,
            command_boot_id TEXT,
            command_sequence INTEGER,
            received_time_ns INTEGER,
            disposition TEXT,
            status_codes_json TEXT,
            detail_json TEXT
        );
        CREATE INDEX idx_actuator_receipts_command
        ON actuator_command_receipts(object_id, command_source_id, command_boot_id, command_sequence);

        CREATE TABLE actuator_realization (
            object_id TEXT,
            actuator_id TEXT,
            interval_start_ns INTEGER,
            interval_end_ns INTEGER,
            command_source_id TEXT,
            command_boot_id TEXT,
            command_sequence INTEGER,
            demand_mode TEXT,
            saturated INTEGER,
            mass_flow_kg_s REAL,
            requested_force_x_n REAL,
            requested_force_y_n REAL,
            requested_force_z_n REAL,
            requested_torque_x_n_m REAL,
            requested_torque_y_n_m REAL,
            requested_torque_z_n_m REAL,
            realized_force_x_n REAL,
            realized_force_y_n REAL,
            realized_force_z_n REAL,
            realized_torque_x_n_m REAL,
            realized_torque_y_n_m REAL,
            realized_torque_z_n_m REAL,
            detail_json TEXT
        );
        CREATE INDEX idx_actuator_realization_command
        ON actuator_realization(object_id, command_source_id, command_boot_id, command_sequence);

        CREATE TABLE actuator_device_state (
            object_id TEXT,
            actuator_id TEXT,
            interval_start_ns INTEGER,
            device_index INTEGER,
            field_name TEXT,
            unit TEXT,
            value_kind TEXT,
            value_real REAL,
            value_integer INTEGER,
            value_text TEXT,
            value_json TEXT
        );
        CREATE INDEX idx_actuator_device_state_object_time
        ON actuator_device_state(object_id, actuator_id, interval_start_ns, field_name);

        CREATE TABLE fsw_diagnostics (
            object_id TEXT,
            invocation_id INTEGER,
            topic TEXT,
            generated_time_ns INTEGER,
            detail_json TEXT
        );

        CREATE TABLE fsw_diagnostic_fields (
            object_id TEXT,
            invocation_id INTEGER,
            topic TEXT,
            generated_time_ns INTEGER,
            field_index INTEGER,
            field_name TEXT,
            unit TEXT,
            value_kind TEXT,
            value_real REAL,
            value_integer INTEGER,
            value_text TEXT,
            value_json TEXT
        );
        CREATE INDEX idx_fsw_diagnostic_fields_object_name_time
        ON fsw_diagnostic_fields(object_id, field_name, generated_time_ns);

        CREATE TABLE safety_requirement_evidence (
            object_id TEXT,
            invocation_id INTEGER,
            requirement_id TEXT,
            satisfied INTEGER,
            source TEXT,
            detail_json TEXT
        );

        CREATE TABLE fsw_snapshots (
            object_id TEXT,
            invocation_id INTEGER,
            stack_id TEXT,
            stack_version TEXT,
            state_hash_sha256 TEXT,
            detail_json TEXT,
            detail_gzip BLOB
        );

        CREATE TABLE game_input_events (
            object_id TEXT,
            invocation_id INTEGER,
            packet_source_id TEXT,
            packet_boot_id TEXT,
            packet_sequence INTEGER,
            input_profile TEXT,
            detail_json TEXT
        );

        CREATE TABLE game_observer_samples (
            object_id TEXT,
            time_ns INTEGER,
            observer_policy TEXT,
            truth_assisted INTEGER,
            detail_json TEXT
        );

        CREATE TABLE game_scoring_events (
            object_id TEXT,
            time_ns INTEGER,
            scoring_policy TEXT,
            event_type TEXT,
            detail_json TEXT
        );

        CREATE TABLE attitude_error (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            desired_quat_w REAL,
            desired_quat_x REAL,
            desired_quat_y REAL,
            desired_quat_z REAL,
            actual_quat_w REAL,
            actual_quat_x REAL,
            actual_quat_y REAL,
            actual_quat_z REAL,
            pointing_error_deg REAL,
            quat_error_angle_deg REAL
        );

        CREATE TABLE ground_access (
            sample_index INTEGER,
            time_s REAL,
            station_id TEXT,
            object_id TEXT,
            access INTEGER,
            line_of_sight INTEGER,
            range_km REAL,
            elevation_deg REAL,
            reason TEXT
        );
        CREATE INDEX idx_ground_access_station_object_time ON ground_access(station_id, object_id, time_s);

        CREATE TABLE ground_access_windows (
            station_id TEXT,
            object_id TEXT,
            window_index INTEGER,
            start_s REAL,
            end_s REAL,
            duration_s REAL,
            start_censored INTEGER,
            end_censored INTEGER,
            minimum_range_km REAL,
            maximum_elevation_deg REAL,
            boundary_semantics TEXT,
            last_access_sample_s REAL,
            first_no_access_sample_s REAL,
            sample_span_s REAL,
            credited_duration_s REAL,
            refinement_status TEXT
        );
        CREATE INDEX idx_ground_access_windows_station_object ON ground_access_windows(station_id, object_id, start_s);

        CREATE TABLE object_orbital_elements (
            sample_index INTEGER,
            time_s REAL,
            object_id TEXT,
            radius_km REAL,
            speed_km_s REAL,
            a_km REAL,
            ecc REAL,
            inc_deg REAL,
            raan_deg REAL,
            argp_deg REAL,
            true_anomaly_deg REAL,
            circular_conditioned INTEGER,
            equatorial_conditioned INTEGER,
            conversion_status TEXT
        );
        CREATE INDEX idx_object_orbital_elements_object_time ON object_orbital_elements(object_id, time_s);

        CREATE TABLE coverage_summary (
            analysis_id TEXT PRIMARY KEY,
            source_object_id TEXT,
            state_provider_id TEXT,
            product_kind TEXT,
            attitude_source_kind TEXT,
            attitude_provider_id TEXT,
            refinement_source TEXT,
            input_evidence_sha256 TEXT,
            semantic_sha256 TEXT,
            summary_json TEXT
        );

        CREATE TABLE coverage_samples (
            analysis_id TEXT,
            sample_index INTEGER,
            time_s REAL,
            covered_cell_count INTEGER,
            instantaneous_covered_fraction REAL
        );
        CREATE INDEX idx_coverage_samples_analysis_time ON coverage_samples(analysis_id, time_s);

        CREATE TABLE coverage_intervals (
            analysis_id TEXT,
            cell_index INTEGER,
            interval_index INTEGER,
            start_s REAL,
            end_s REAL,
            duration_s REAL,
            start_censored INTEGER,
            end_censored INTEGER,
            acquisition_disposition TEXT,
            loss_disposition TEXT,
            acquisition_reason TEXT,
            loss_reason TEXT
        );
        CREATE INDEX idx_coverage_intervals_analysis_cell ON coverage_intervals(analysis_id, cell_index, start_s);

        CREATE TABLE coverage_transitions (
            analysis_id TEXT,
            cell_index INTEGER,
            transition_kind TEXT,
            time_s REAL,
            bracket_start_s REAL,
            bracket_end_s REAL,
            disposition TEXT,
            iterations INTEGER,
            reason_before TEXT,
            reason_after TEXT
        );
        CREATE INDEX idx_coverage_transitions_analysis_cell ON coverage_transitions(analysis_id, cell_index, time_s);

        CREATE TABLE link_summary (
            analysis_id TEXT PRIMARY KEY,
            link_id TEXT,
            tx_object_id TEXT,
            rx_object_id TEXT,
            tx_endpoint_kind TEXT,
            rx_endpoint_kind TEXT,
            tx_terminal_parent_frame TEXT,
            rx_terminal_parent_frame TEXT,
            tx_state_provider_id TEXT,
            rx_state_provider_id TEXT,
            tx_attitude_source_kind TEXT,
            tx_attitude_provider_id TEXT,
            rx_attitude_source_kind TEXT,
            rx_attitude_provider_id TEXT,
            refinement_source_json TEXT,
            input_evidence_sha256 TEXT,
            semantic_sha256 TEXT,
            summary_json TEXT
        );

        CREATE TABLE link_samples (
            analysis_id TEXT,
            sample_index INTEGER,
            time_s REAL,
            range_km REAL,
            margin_db REAL,
            available INTEGER,
            primary_reason TEXT
        );
        CREATE INDEX idx_link_samples_analysis_time ON link_samples(analysis_id, time_s);

        CREATE TABLE link_windows (
            analysis_id TEXT,
            interval_index INTEGER,
            start_s REAL,
            end_s REAL,
            duration_s REAL,
            start_censored INTEGER,
            end_censored INTEGER,
            acquisition_disposition TEXT,
            loss_disposition TEXT,
            minimum_margin_db REAL,
            mean_margin_db REAL,
            maximum_margin_db REAL,
            minimum_range_km REAL,
            estimated_delivered_data_bits REAL
        );

        CREATE TABLE link_transitions (
            analysis_id TEXT,
            transition_kind TEXT,
            time_s REAL,
            bracket_start_s REAL,
            bracket_end_s REAL,
            disposition TEXT,
            iterations INTEGER,
            reason_before TEXT,
            reason_after TEXT
        );

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            time_s REAL,
            sample_index INTEGER,
            object_id TEXT,
            event_type TEXT,
            severity TEXT,
            message TEXT,
            source TEXT
        );

        CREATE TABLE metrics (
            metric_id TEXT PRIMARY KEY,
            metric_name TEXT,
            object_id TEXT,
            deputy_id TEXT,
            chief_id TEXT,
            value REAL,
            units TEXT,
            source TEXT
        );

        CREATE TABLE mission_recovery_summary (
            object_id TEXT,
            goal TEXT,
            method TEXT,
            assessment_time_s REAL,
            assessment_sample_index INTEGER,
            recovery_available INTEGER,
            recovery_delta_v_m_s REAL,
            recovery_time_s REAL,
            recovery_time_basis TEXT,
            propellant_kg REAL,
            propellant_fraction REAL,
            disturbance_delta_v_m_s REAL,
            disturbance_apsis TEXT,
            slot_recovery_found INTEGER,
            slot_recovery_orbits INTEGER,
            slot_recovery_time_s REAL,
            slot_recovery_phase_error_deg REAL,
            best_slot_orbits INTEGER,
            best_slot_time_s REAL,
            best_slot_phase_error_deg REAL,
            local_orbit_shape_delta_v_m_s REAL,
            local_orbit_shape_position_error_km REAL,
            notes_json TEXT
        );

        CREATE TABLE mission_recovery_elements (
            object_id TEXT,
            state_label TEXT,
            a_km REAL,
            ecc REAL,
            inc_deg REAL,
            raan_deg REAL,
            argp_deg REAL,
            true_anomaly_deg REAL
        );

        CREATE TABLE mission_recovery_candidates (
            candidate_id TEXT,
            object_id TEXT,
            goal TEXT,
            source TEXT,
            source_family TEXT,
            target_basis TEXT,
            description TEXT,
            planned_delta_v_m_s REAL,
            simulated_delta_v_m_s REAL,
            planned_time_s REAL,
            simulated_recovery_time_s REAL,
            propellant_kg REAL,
            propellant_fraction REAL,
            feasible INTEGER,
            verified INTEGER,
            within_tolerances INTEGER,
            score REAL,
            recommended_modes_json TEXT,
            transfer_type TEXT,
            departure_wait_s REAL,
            time_of_flight_s REAL,
            arrival_time_s REAL,
            target_phase_deg REAL,
            lambert_short_way INTEGER,
            lambert_revolutions INTEGER,
            solver_iterations INTEGER,
            solver_residual_s REAL,
            position_residual_km REAL,
            velocity_residual_m_s REAL,
            notes_json TEXT
        );

        CREATE TABLE mission_recovery_burns (
            candidate_id TEXT,
            burn_index INTEGER,
            start_time_s REAL,
            duration_s REAL,
            frame TEXT,
            axis TEXT,
            delta_v_m_s REAL,
            delta_v_eci_m_s_json TEXT
        );

        CREATE TABLE mission_recovery_candidate_elements (
            candidate_id TEXT,
            object_id TEXT,
            a_km REAL,
            ecc REAL,
            inc_deg REAL,
            raan_deg REAL,
            argp_deg REAL,
            true_anomaly_deg REAL,
            element_errors_json TEXT
        );

        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            artifact_type TEXT,
            path TEXT,
            title TEXT,
            source TEXT,
            created_utc TEXT
        );
        COMMIT;
        """
    )
    try:
        conn.executescript(schema_sql)
    except BaseException:
        # executescript does not roll back an explicit failed transaction.
        # Preserve its original exception even for an already closed handle.
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        raise


def _insert_run_metadata(
    conn: sqlite3.Connection,
    *,
    cfg: Any,
    summary: dict[str, Any],
    outdir: Path,
    generated_utc: str,
    config_json: str | None = None,
    config_sha256: str | None = None,
) -> None:
    canonical_config_json, canonical_config_sha256 = _config_json_and_sha256(cfg)
    if config_json is not None and config_json != canonical_config_json:
        raise ValueError("Precomputed config_json does not match the scenario configuration.")
    if config_sha256 is not None and config_sha256 != canonical_config_sha256:
        raise ValueError("Precomputed config_sha256 does not match the scenario configuration.")
    config_json = canonical_config_json
    config_sha256 = canonical_config_sha256
    conn.execute(
        """
        INSERT INTO run_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(summary.get("scenario_name") or cfg.scenario_name or "single_run"),
            str(summary.get("scenario_name") or cfg.scenario_name or ""),
            str(summary.get("scenario_description") or cfg.scenario_description or ""),
            get_oel_version(),
            REVIEW_SCHEMA_VERSION,
            generated_utc,
            _float_or_none(summary.get("duration_s")),
            _float_or_none(summary.get("dt_s")),
            _int_or_none(summary.get("samples")),
            str(outdir),
            str(summary.get("config_source_path") or ""),
            config_sha256,
            config_json,
            str(outdir / "master_run_summary.json"),
            str(outdir / "master_run_log.json"),
        ),
    )


def _config_json_and_sha256(cfg: Any) -> tuple[str, str]:
    if hasattr(cfg, "to_dict"):
        data = cfg.to_dict()
    else:
        data = {}
    config_json = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _insert_objects(conn: sqlite3.Connection, *, cfg: Any, summary: dict[str, Any]) -> None:
    sections = {str(object_id): section for object_id, section in iter_object_sections(cfg)}
    object_ids = [str(item) for item in list(summary.get("objects", []) or [])]
    rows = []
    for object_id in object_ids:
        section = sections.get(object_id)
        specs = dict(getattr(section, "specs", {}) or {}) if section is not None else {}
        rows.append(
            (
                object_id,
                str(getattr(section, "kind", "") or ""),
                int(bool(getattr(section, "enabled", True))),
                _float_or_none(specs.get("mass_kg", specs.get("dry_mass_kg"))),
                str(getattr(section, "role", "") or ""),
                str(getattr(section, "runtime_profile", "flight_software") or "flight_software"),
                str(getattr(getattr(section, "flight_software", None), "stack", "") or ""),
            )
        )
    conn.executemany("INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_frame_provenance(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    frame = dict(payload.get("frame_provenance", {}) or {})
    if not frame:
        frame = dict(dict(payload.get("summary", {}) or {}).get("frame_provenance", {}) or {})
    if not frame:
        return
    row = (
        "scenario",
        str(frame.get("model", "") or ""),
        str(frame.get("legacy_frame_model", "") or ""),
        str(frame.get("time_scale_model", "") or ""),
        str(frame.get("eop_path", "") or ""),
        _float_or_none(frame.get("tt_minus_utc_s")),
        _float_or_none(frame.get("dut1_s")),
        _float_or_none(frame.get("xp_arcsec")),
        _float_or_none(frame.get("yp_arcsec")),
        _float_or_none(frame.get("dat_s")),
        _float_or_none(frame.get("ddpsi_rad")),
        _float_or_none(frame.get("ddeps_rad")),
        1 if bool(frame.get("polar_motion_applied", False)) else 0,
        1 if bool(frame.get("nutation_corrections_applied", False)) else 0,
        _float_or_none(frame.get("sample_t_s")),
    )
    conn.execute("INSERT INTO frame_provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)


def _insert_object_initialization(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    rows = []
    for object_id, metadata_raw in sorted(dict(payload.get("object_initialization", {}) or {}).items()):
        metadata = dict(metadata_raw or {})
        rows.append(
            (
                str(object_id),
                str(metadata.get("source", "") or ""),
                str(metadata.get("initialization_model", "") or ""),
                str(metadata.get("initialization_propagator_family", "") or ""),
                str(metadata.get("initialization_propagator_name", "") or ""),
                str(metadata.get("handoff_propagation_method", "") or ""),
                str(metadata.get("native_frame", "") or ""),
                str(metadata.get("output_frame", "") or ""),
                str(metadata.get("frame_transform", "") or ""),
                _float_or_none(metadata.get("tle_epoch_jd_utc")),
                _float_or_none(metadata.get("initial_jd_utc")),
                _float_or_none(metadata.get("tle_age_initialization_days")),
                1 if bool(metadata.get("propagate_to_initial_epoch", False)) else 0,
                _float_or_none(metadata.get("simulation_duration_s")),
                str(metadata.get("note", "") or ""),
            )
        )
    conn.executemany(
        "INSERT INTO object_initialization VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _insert_object_propagation(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    rows = []
    for object_id, metadata_raw in sorted(dict(payload.get("object_propagation", {}) or {}).items()):
        metadata = dict(metadata_raw or {})
        rows.append(
            (
                str(object_id),
                str(metadata.get("propagation_method", "") or ""),
                str(metadata.get("propagator_family", "") or ""),
                str(metadata.get("propagator_name", "") or ""),
                str(metadata.get("general_model", "") or ""),
                str(metadata.get("native_frame", "") or ""),
                str(metadata.get("output_frame", "") or ""),
                str(metadata.get("state_history_frame", "") or ""),
                str(metadata.get("frame_transform", "") or ""),
                _float_or_none(metadata.get("tle_epoch_jd_utc")),
                _float_or_none(metadata.get("tle_age_start_days")),
                _float_or_none(metadata.get("tle_age_end_days")),
                str(metadata.get("ogp_regime", "") or ""),
                _float_or_none(metadata.get("orbital_period_min")),
                _float_or_none(metadata.get("max_tle_age_days_warning")),
                _bool_int(metadata.get("tle_age_warning")),
            )
        )
    conn.executemany(
        "INSERT INTO object_propagation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _insert_object_state_frame(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    explicit = dict(payload.get("object_state_frames", {}) or {})
    propagation = dict(payload.get("object_propagation", {}) or {})
    object_ids = sorted(set(str(key) for key in explicit) | set(str(key) for key in propagation))
    rows = []
    for object_id in object_ids:
        frame = explicit.get(object_id)
        if frame is None:
            frame = dict(propagation.get(object_id, {}) or {}).get("state_history_frame", "eci")
        rows.append((object_id, str(frame or "eci").strip().lower()))
    conn.executemany("INSERT OR REPLACE INTO object_state_frame VALUES (?, ?)", rows)


def _insert_time_samples(conn: sqlite3.Connection, *, t_s: np.ndarray) -> None:
    conn.executemany("INSERT INTO time_samples VALUES (?, ?)", [(int(i), _finite_float(t)) for i, t in enumerate(t_s)])


def _insert_object_state(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
) -> None:
    rows = []
    for object_id, hist in truth_hist.items():
        n = int(min(t_s.size, hist.shape[0]))
        for i in range(n):
            row = hist[i]
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    object_id,
                    _state_value(row, 0),
                    _state_value(row, 1),
                    _state_value(row, 2),
                    _state_value(row, 3),
                    _state_value(row, 4),
                    _state_value(row, 5),
                    _state_value(row, 6),
                    _state_value(row, 7),
                    _state_value(row, 8),
                    _state_value(row, 9),
                    _state_value(row, 10),
                    _state_value(row, 11),
                    _state_value(row, 12),
                    _state_value(row, 13),
                )
            )
    conn.executemany(
        """
        INSERT INTO object_state VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        rows,
    )


def _insert_object_orbital_elements(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    object_state_frames: dict[str, str],
) -> None:
    rows = []
    for object_id, hist in truth_hist.items():
        if str(object_state_frames.get(object_id, "eci") or "eci").strip().lower() != "eci" or hist.shape[1] < 6:
            continue
        n = int(min(t_s.size, hist.shape[0]))
        batched = _batched_orbital_elements(np.asarray(hist[:n, :6], dtype=float))
        for i in range(n):
            state = np.asarray(hist[i, :6], dtype=float)
            radius = float(np.linalg.norm(state[:3])) if np.all(np.isfinite(state[:3])) else None
            speed = float(np.linalg.norm(state[3:6])) if np.all(np.isfinite(state[3:6])) else None
            values: tuple[Any, ...] = (None,) * 6
            circular = None
            equatorial = None
            status = "unavailable"
            if np.all(np.isfinite(state)):
                batch_values = batched[i]
                if batch_values is not None:
                    values = batch_values
                    circular = int(values[1] <= 1.0e-8)
                    equatorial = int(min(abs(values[2]), abs(180.0 - values[2])) <= 1.0e-8)
                    status = "conditioned" if circular or equatorial else "ok"
                else:
                    # Preserve the scalar implementation as the authoritative
                    # edge-case/error path for degenerate or unsupported states.
                    try:
                        coes = rv_to_coe_eci(state[:3], state[3:6])
                        circular = int(coes.ecc <= 1.0e-8)
                        equatorial = int(min(abs(coes.inc_deg), abs(180.0 - coes.inc_deg)) <= 1.0e-8)
                        values = (
                            coes.a_km,
                            coes.ecc,
                            coes.inc_deg,
                            coes.raan_deg,
                            coes.argp_deg,
                            coes.true_anomaly_deg,
                        )
                        status = "conditioned" if circular or equatorial else "ok"
                    except ValueError as exc:
                        status = f"unsupported:{type(exc).__name__}"
            rows.append(
                (i, _finite_float(t_s[i]), object_id, radius, speed, *values, circular, equatorial, status)
            )
    conn.executemany(
        "INSERT INTO object_orbital_elements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _batched_orbital_elements(states: np.ndarray) -> list[tuple[float, ...] | None]:
    """Evaluate ordinary elliptical ECI states in bulk with scalar-equivalent math."""

    x = np.asarray(states, dtype=float)
    count = int(x.shape[0]) if x.ndim == 2 and x.shape[1] >= 6 else 0
    output: list[tuple[float, ...] | None] = [None] * count
    if count == 0:
        return output
    finite = np.all(np.isfinite(x[:, :6]), axis=1)
    if not bool(np.any(finite)):
        return output
    indices = np.flatnonzero(finite)
    values = x[indices, :6]
    r = values[:, :3]
    v = values[:, 3:6]
    mu = EARTH_MU_KM3_S2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        radius = np.linalg.norm(r, axis=1)
        speed = np.linalg.norm(v, axis=1)
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h, axis=1)
        k_hat = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=float), h.shape)
        node = np.cross(k_hat, h)
        node_norm = np.linalg.norm(node, axis=1)
        eccentricity_vector = np.cross(v, h) / mu - r / radius[:, None]
        eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
        energy = 0.5 * speed * speed - mu / radius
        semi_major_axis = -mu / (2.0 * energy)
    ordinary = (
        (radius > 0.0)
        & (h_norm > 0.0)
        & (np.abs(energy) > 1.0e-15)
        & (semi_major_axis > 0.0)
        & (eccentricity < 1.0)
        & np.all(
            np.isfinite(
                np.column_stack(
                    (radius, speed, h_norm, node_norm, eccentricity, energy, semi_major_axis)
                )
            ),
            axis=1,
        )
    )
    if not bool(np.any(ordinary)):
        return output

    with np.errstate(divide="ignore", invalid="ignore"):
        inclination = np.arccos(np.clip(h[:, 2] / h_norm, -1.0, 1.0))
    raan = np.zeros(indices.size, dtype=float)
    node_mask = node_norm > 1.0e-12
    raan[node_mask] = np.arccos(np.clip(node[node_mask, 0] / node_norm[node_mask], -1.0, 1.0))
    raan[node_mask & (node[:, 1] < 0.0)] = 2.0 * np.pi - raan[node_mask & (node[:, 1] < 0.0)]

    argp = np.zeros(indices.size, dtype=float)
    argp_mask = node_mask & (eccentricity > 1.0e-10)
    argp[argp_mask] = np.arccos(
        np.clip(
            np.sum(node[argp_mask] * eccentricity_vector[argp_mask], axis=1)
            / (node_norm[argp_mask] * eccentricity[argp_mask]),
            -1.0,
            1.0,
        )
    )
    argp[argp_mask & (eccentricity_vector[:, 2] < 0.0)] = (
        2.0 * np.pi - argp[argp_mask & (eccentricity_vector[:, 2] < 0.0)]
    )

    true_anomaly = np.zeros(indices.size, dtype=float)
    eccentric_mask = eccentricity > 1.0e-10
    true_anomaly[eccentric_mask] = np.arccos(
        np.clip(
            np.sum(eccentricity_vector[eccentric_mask] * r[eccentric_mask], axis=1)
            / (eccentricity[eccentric_mask] * radius[eccentric_mask]),
            -1.0,
            1.0,
        )
    )
    radial_velocity = np.sum(r * v, axis=1)
    true_anomaly[eccentric_mask & (radial_velocity < 0.0)] = (
        2.0 * np.pi - true_anomaly[eccentric_mask & (radial_velocity < 0.0)]
    )
    circular_inclined = (~eccentric_mask) & node_mask
    true_anomaly[circular_inclined] = np.arccos(
        np.clip(
            np.sum(node[circular_inclined] * r[circular_inclined], axis=1)
            / (node_norm[circular_inclined] * radius[circular_inclined]),
            -1.0,
            1.0,
        )
    )
    true_anomaly[circular_inclined & (r[:, 2] < 0.0)] = (
        2.0 * np.pi - true_anomaly[circular_inclined & (r[:, 2] < 0.0)]
    )
    circular_equatorial = (~eccentric_mask) & (~node_mask)
    true_anomaly[circular_equatorial] = np.arctan2(
        r[circular_equatorial, 1], r[circular_equatorial, 0]
    )

    def degrees(angle: np.ndarray) -> np.ndarray:
        return np.rad2deg(np.mod(angle, 2.0 * np.pi))
    derived = np.column_stack(
        (
            semi_major_axis,
            eccentricity,
            degrees(inclination),
            degrees(raan),
            degrees(argp),
            degrees(true_anomaly),
        )
    )
    ordinary &= np.all(np.isfinite(derived), axis=1)
    for local_index in np.flatnonzero(ordinary):
        if node_norm[local_index] <= 1.0e-12 * h_norm[local_index]:
            # Keep singular-plane conventions in the authoritative scalar owner,
            # including eccentric and retrograde equatorial orientations.
            coes = rv_to_coe_eci(r[local_index], v[local_index])
            output[int(indices[local_index])] = (
                coes.a_km, coes.ecc, coes.inc_deg, coes.raan_deg,
                coes.argp_deg, coes.true_anomaly_deg,
            )
        else:
            output[int(indices[local_index])] = tuple(float(item) for item in derived[local_index])
    return output


def _insert_attitude_error(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    payload: dict[str, Any],
) -> None:
    desired_root = dict(payload.get("desired_attitude_by_object", {}) or {})
    rows = []
    for object_id, desired_raw in desired_root.items():
        actual = truth_hist.get(str(object_id))
        desired = np.asarray(desired_raw, dtype=float)
        if actual is None or actual.ndim != 2 or actual.shape[1] < 10 or desired.ndim != 2 or desired.shape[1] < 4:
            continue
        n = int(min(t_s.size, actual.shape[0], desired.shape[0]))
        for i in range(n):
            q_actual = np.asarray(actual[i, 6:10], dtype=float)
            q_desired = np.asarray(desired[i, :4], dtype=float)
            if not np.all(np.isfinite(q_actual)) or not np.all(np.isfinite(q_desired)):
                continue
            actual_norm = float(np.linalg.norm(q_actual))
            desired_norm = float(np.linalg.norm(q_desired))
            if actual_norm <= 0.0 or desired_norm <= 0.0:
                continue
            q_actual = q_actual / actual_norm
            q_desired = q_desired / desired_norm
            angle_deg = float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(q_actual, q_desired))), 0.0, 1.0))))
            rows.append(
                (i, _finite_float(t_s[i]), str(object_id), *q_desired.tolist(), *q_actual.tolist(), angle_deg, angle_deg)
            )
    populated = {str(row[2]) for row in rows}
    fsw_root = dict(payload.get("flight_software_evidence_by_object", {}) or {})
    for object_id, evidence_raw in fsw_root.items():
        if str(object_id) in populated:
            continue
        actual = truth_hist.get(str(object_id))
        if actual is None or actual.ndim != 2 or actual.shape[1] < 10 or t_s.size == 0:
            continue
        for output_raw in list(dict(evidence_raw or {}).get("outputs", []) or []):
            for telemetry_raw in list(dict(output_raw or {}).get("telemetry", []) or []):
                telemetry = dict(telemetry_raw or {})
                generated = dict(telemetry.get("generated_at", {}) or {})
                ticks = _float_or_none(generated.get("ticks"))
                tick_period_ns = _float_or_none(generated.get("tick_period_ns"))
                if ticks is None or tick_period_ns is None:
                    continue
                time_s = ticks * tick_period_ns * 1.0e-9
                error_rad = None
                desired_components: dict[str, float] = {}
                for field_raw in list(telemetry.get("fields", []) or []):
                    field = dict(field_raw or {})
                    field_name = str(field.get("name") or "")
                    if field_name == "attitude_error_rad":
                        error_rad = _float_or_none(field.get("value"))
                    elif field_name in {
                        "desired_attitude_quat_w",
                        "desired_attitude_quat_x",
                        "desired_attitude_quat_y",
                        "desired_attitude_quat_z",
                    }:
                        component = _float_or_none(field.get("value"))
                        if component is not None:
                            desired_components[field_name[-1]] = component
                if error_rad is None:
                    continue
                sample_index = int(np.argmin(np.abs(t_s - time_s)))
                q_actual = np.asarray(actual[min(sample_index, actual.shape[0] - 1), 6:10], dtype=float)
                if not np.all(np.isfinite(q_actual)) or float(np.linalg.norm(q_actual)) <= 0.0:
                    actual_values = (None, None, None, None)
                else:
                    actual_values = tuple((q_actual / float(np.linalg.norm(q_actual))).tolist())
                angle_deg = float(np.degrees(abs(error_rad)))
                desired_values: tuple[float | None, ...]
                if set(desired_components) == {"w", "x", "y", "z"}:
                    q_desired = np.asarray(
                        [desired_components[key] for key in ("w", "x", "y", "z")],
                        dtype=float,
                    )
                    norm = float(np.linalg.norm(q_desired))
                    desired_values = (
                        tuple((q_desired / norm).tolist())
                        if norm > 0.0
                        else (None, None, None, None)
                    )
                else:
                    desired_values = (None, None, None, None)
                rows.append(
                    (
                        sample_index,
                        time_s,
                        str(object_id),
                        *desired_values,
                        *actual_values,
                        angle_deg,
                        angle_deg,
                    )
                )
    conn.executemany(
        "INSERT INTO attitude_error VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )


def _insert_relative_state(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    summary: dict[str, Any],
    cfg: Any,
    object_state_frames: dict[str, str] | None = None,
) -> None:
    rows = []
    state_frames = dict(object_state_frames or {})
    for deputy_id, chief_id in _relative_review_pairs(cfg=cfg, truth_hist=truth_hist, summary=summary):
        deputy_frame = str(state_frames.get(deputy_id, "eci") or "eci").strip().lower()
        chief_frame = str(state_frames.get(chief_id, "eci") or "eci").strip().lower()
        if deputy_frame != "eci" or chief_frame != "eci":
            continue
        deputy = truth_hist.get(deputy_id)
        chief = truth_hist.get(chief_id)
        if deputy is None or chief is None or deputy.shape[1] < 6 or chief.shape[1] < 6:
            continue
        n = int(min(t_s.size, deputy.shape[0], chief.shape[0]))
        for i in range(n):
            rel = (
                eci_relative_to_ric_rect(deputy[i, :6], chief[i, :6])
                if np.isfinite(deputy[i, :6]).all() and np.isfinite(chief[i, :6]).all()
                else np.full(6, np.nan)
            )
            rng = float(np.linalg.norm(rel[:3]))
            range_rate = (
                float(np.dot(rel[:3], rel[3:]) / rng)
                if math.isfinite(rng) and rng > 1e-12 and bool(np.all(np.isfinite(rel[:6])))
                else None
            )
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    deputy_id,
                    chief_id,
                    _finite_float(rel[0]),
                    _finite_float(rel[1]),
                    _finite_float(rel[2]),
                    _finite_float(rel[3]),
                    _finite_float(rel[4]),
                    _finite_float(rel[5]),
                    _finite_float(rng),
                    _finite_float(range_rate),
                )
            )
    conn.executemany("INSERT INTO relative_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _relative_review_pairs(
    *,
    cfg: Any,
    truth_hist: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    def add_pair(deputy_id: str, chief_id: str) -> None:
        pair = (str(deputy_id), str(chief_id))
        if pair[0] == pair[1] or pair[0] not in truth_hist or pair[1] not in truth_hist:
            return
        if pair not in pairs:
            pairs.append(pair)

    primary = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    if len(primary) == 2:
        add_pair(primary[0], primary[1])

    for object_id, _section in iter_object_sections(cfg, enabled_only=True):
        reference_id = relative_reference_for_object(cfg, object_id)
        if reference_id:
            add_pair(object_id, reference_id)

    return pairs


def _insert_thrust(conn: sqlite3.Connection, *, t_s: np.ndarray, thrust_hist: dict[str, np.ndarray]) -> None:
    rows = []
    for object_id, hist in thrust_hist.items():
        if hist.shape[1] < 3:
            continue
        n = int(min(t_s.size, hist.shape[0]))
        for i in range(n):
            vec = np.asarray(hist[i, :3], dtype=float)
            norm = float(np.linalg.norm(vec))
            rows.append(
                (
                    i,
                    _finite_float(t_s[i]),
                    object_id,
                    _finite_float(vec[0]),
                    _finite_float(vec[1]),
                    _finite_float(vec[2]),
                    _finite_float(norm),
                    int(norm > 0.0),
                )
            )
    conn.executemany("INSERT INTO thrust VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_command_decisions(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    include_debug_detail: bool = True,
) -> None:
    decision_rows = []
    mode_rows = []
    transition_rows = []
    gate_rows = []
    decisions_by_object = dict(payload.get("command_decisions_by_object", {}) or {})
    for object_id, raw_rows in sorted(decisions_by_object.items()):
        for decision_index, raw_row in enumerate(list(raw_rows or [])):
            row = dict(raw_row or {})
            mission_mode = dict(row.get("mission_mode", {}) or {})
            detail_json = _debug_json(row, include=include_debug_detail)
            common = (
                int(decision_index),
                _int_or_none(row.get("sample_index")),
                _float_or_none(row.get("time_s")),
                str(object_id),
            )
            decision_rows.append(
                (
                    int(decision_index),
                    _int_or_none(row.get("sample_index")),
                    _float_or_none(row.get("time_s")),
                    _float_or_none(row.get("interval_end_time_s")),
                    _float_or_none(row.get("dt_s")),
                    str(object_id),
                    _none_or_str(row.get("orbit_controller")),
                    _none_or_str(row.get("attitude_controller")),
                    _none_or_str(row.get("mission_strategy")),
                    _none_or_str(row.get("mission_execution")),
                    _float_or_none(row.get("requested_accel_norm_km_s2")),
                    _float_or_none(row.get("applied_accel_norm_km_s2")),
                    _float_or_none(row.get("requested_torque_norm_nm")),
                    _float_or_none(row.get("applied_torque_norm_nm")),
                    _bool_int(row.get("burn_requested")),
                    _bool_int(row.get("burn_applied")),
                    _bool_int(row.get("saturated")),
                    _bool_int(row.get("deadline_missed")),
                    json.dumps(dict(row.get("field_sources", {}) or {}), sort_keys=True),
                    json.dumps(list(row.get("collisions", []) or []), sort_keys=True),
                    detail_json,
                )
            )
            mode_rows.append(
                (
                    *common,
                    _none_or_str(row.get("mission_strategy")),
                    _none_or_str(row.get("mission_execution")),
                    _none_or_str(row.get("mission_phase")),
                    _none_or_str(row.get("executive_mode")),
                    json.dumps(mission_mode, sort_keys=True, separators=(",", ":")),
                )
            )
            transition = dict(mission_mode.get("executive_transition", {}) or {})
            if transition:
                detail = transition.get("detail")
                reason = None
                if isinstance(detail, dict):
                    reason = detail.get("reason")
                elif detail is not None:
                    reason = str(detail)
                transition_rows.append(
                    (
                        *common,
                        _none_or_str(transition.get("from_mode")),
                        _none_or_str(transition.get("to_mode")),
                        _none_or_str(transition.get("trigger")),
                        _none_or_str(reason),
                        json.dumps(transition, sort_keys=True, separators=(",", ":")),
                    )
                )
            gate_rows.append(
                (
                    *common,
                    _bool_int(row.get("burn_requested")),
                    _bool_int(row.get("burn_applied")),
                    _float_or_none(row.get("alignment_error_rad")),
                    _bool_int(row.get("alignment_ok")),
                    _bool_int(row.get("fuel_depleted")),
                    _bool_int(row.get("actuator_limited")),
                    _bool_int(row.get("deadline_missed")),
                    _none_or_str(row.get("gate_reason")),
                    None,
                )
            )
    conn.executemany(
        "INSERT INTO controller_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        decision_rows,
    )
    conn.executemany("INSERT INTO mission_modes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", mode_rows)
    conn.executemany("INSERT INTO mission_transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", transition_rows)
    conn.executemany("INSERT INTO command_gates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", gate_rows)


def _packet_key(packet: Any) -> tuple[str, str, int | None]:
    value = dict(packet or {})
    return (
        str(value.get("source_id", "") or ""),
        str(value.get("boot_id", "") or ""),
        _int_or_none(value.get("sequence")),
    )


def _clock_ns(clock: Any) -> int | None:
    value = dict(clock or {})
    ticks = _int_or_none(value.get("ticks"))
    tick_period_ns = _int_or_none(value.get("tick_period_ns"))
    if ticks is None or tick_period_ns is None:
        return None
    return ticks * tick_period_ns


def _debug_json(value: Any, *, include: bool) -> str | None:
    if not include:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _typed_review_value(value: Any) -> tuple[str, float | None, int | None, str | None, str | None]:
    if value is None:
        return "null", None, None, None, None
    if isinstance(value, (bool, np.bool_)):
        return "boolean", None, int(bool(value)), None, None
    if isinstance(value, (int, np.integer)):
        return "integer", None, int(value), None, None
    if isinstance(value, (float, np.floating)):
        return "real", _finite_float(value), None, None, None
    if isinstance(value, str):
        return "text", None, None, value, None
    return "json", None, None, None, json.dumps(value, sort_keys=True, separators=(",", ":"))


def _vector3(value: Any) -> tuple[float | None, float | None, float | None]:
    try:
        vector = list(value or [])
    except TypeError:
        vector = []
    return tuple(_float_or_none(vector[index]) if index < len(vector) else None for index in range(3))  # type: ignore[return-value]


def _standard_task_release_detail(release: dict[str, Any], *, include_debug_detail: bool) -> str:
    if include_debug_detail:
        return json.dumps(release, sort_keys=True, separators=(",", ":"))
    return json.dumps(
        {"release_reasons": list(release.get("release_reasons", []) or [])},
        sort_keys=True,
        separators=(",", ":"),
    )


def _snapshot_storage(snapshot: dict[str, Any], *, include_debug_detail: bool) -> tuple[str | None, bytes | None]:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if include_debug_detail:
        return encoded.decode("utf-8"), None
    return None, gzip.compress(encoded, compresslevel=1, mtime=0)


def _mission_load_dispositions(evidence: dict[str, Any]) -> dict[tuple[int, str, int | None], str]:
    """Index stack-owned load decisions without treating delivery as activation."""

    results: dict[tuple[int, str, int | None], str] = {}
    for raw_output in list(evidence.get("outputs", []) or []):
        output = dict(raw_output or {})
        invocation_id = int(output.get("invocation_id", 0) or 0)
        for raw_telemetry in list(output.get("telemetry", []) or []):
            fields = {
                str(item.get("name")): item.get("value")
                for item in list(dict(raw_telemetry or {}).get("fields", []) or [])
                if isinstance(item, dict) and item.get("name") is not None
            }
            if fields.get("mission_load_id") is None:
                continue
            results[
                (
                    invocation_id,
                    str(fields.get("mission_load_id")),
                    _int_or_none(fields.get("mission_load_revision")),
                )
            ] = str(fields.get("mission_load_disposition", "delivery_unresolved") or "delivery_unresolved")
    return results


def _insert_flight_software_evidence(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
    include_debug_detail: bool = True,
) -> None:
    evidence_by_object = dict(payload.get("flight_software_evidence_by_object", {}) or {})
    invocation_rows = []
    input_rows = []
    load_rows = []
    objective_rows = []
    task_rows = []
    command_rows = []
    receipt_rows = []
    realization_rows = []
    device_state_rows = []
    diagnostic_rows = []
    diagnostic_field_rows = []
    safety_rows = []
    snapshot_rows = []
    game_input_rows = []

    for object_id, raw_evidence in sorted(evidence_by_object.items()):
        evidence = dict(raw_evidence or {})
        load_dispositions = _mission_load_dispositions(evidence)
        packet_invocations: dict[tuple[str, str, int | None], int] = {}
        for raw_invocation in list(evidence.get("invocations", []) or []):
            invocation = dict(raw_invocation or {})
            invocation_id = int(invocation.get("invocation_id", 0) or 0)
            packet_ids = list(invocation.get("input_packet_ids", []) or [])
            command_ids = list(invocation.get("command_ids", []) or [])
            for packet_id in packet_ids:
                packet_invocations[_packet_key(packet_id)] = invocation_id
            invocation_rows.append(
                (
                    str(object_id),
                    invocation_id,
                    _int_or_none(invocation.get("invocation_time_ns")),
                    _none_or_str(invocation.get("stack_id")),
                    _none_or_str(invocation.get("stack_version")),
                    _none_or_str(invocation.get("profile_id")),
                    len(packet_ids),
                    len(command_ids),
                    int(invocation.get("telemetry_count", 0) or 0),
                    _debug_json(invocation, include=include_debug_detail),
                )
            )
            for raw_request in list(invocation.get("requested_next_invocations", []) or []):
                request = dict(raw_request or {})
                task_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        _none_or_str(request.get("task_id")),
                        _clock_ns(request.get("release_at")),
                        None,
                        None,
                        None,
                        _debug_json(request, include=include_debug_detail),
                    )
                )
            for raw_release in list(invocation.get("task_releases", []) or []):
                release = dict(raw_release or {})
                task_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        _none_or_str(release.get("task_id")),
                        _int_or_none(release.get("release_time_ns")),
                        _int_or_none(release.get("modeled_execution_duration_ns")),
                        _int_or_none(release.get("execution_budget_ns")),
                        int(bool(release.get("deadline_missed", False))),
                        _standard_task_release_detail(release, include_debug_detail=include_debug_detail),
                    )
                )

        mission_events: list[dict[str, Any]] = []
        for raw_event in list(evidence.get("input_events", []) or []):
            event = dict(raw_event or {})
            if str(event.get("kind", "")) == "mission_load":
                mission_events.append(event)
        accepted_activation_times_ns: list[int] = []
        for event in mission_events:
            manifest = dict(dict(event.get("payload", {}) or {}).get("manifest", {}) or {})
            invocation_id = packet_invocations.get(_packet_key(event.get("packet_id")))
            disposition = load_dispositions.get(
                (
                    int(invocation_id or 0),
                    str(manifest.get("load_id", "") or ""),
                    _int_or_none(manifest.get("revision")),
                ),
                "delivery_unresolved",
            )
            delivery_ns = _clock_ns(event.get("delivery_time"))
            if disposition == "accepted" and delivery_ns is not None:
                accepted_activation_times_ns.append(delivery_ns)
        accepted_activation_times_ns.sort()

        for raw_event in list(evidence.get("input_events", []) or []):
            event = dict(raw_event or {})
            packet = _packet_key(event.get("packet_id"))
            invocation_id = packet_invocations.get(packet)
            detail = _debug_json(event, include=include_debug_detail)
            input_rows.append(
                (
                    str(object_id),
                    *packet,
                    invocation_id,
                    _none_or_str(event.get("kind")),
                    _clock_ns(event.get("source_time")),
                    _clock_ns(event.get("delivery_time")),
                    _none_or_str(event.get("schema")),
                    detail,
                )
            )
            if str(event.get("kind", "")) == "pilot_input":
                event_payload = dict(event.get("payload", {}) or {})
                game_input_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        *packet,
                        _none_or_str(event_payload.get("input_profile_id")),
                        detail,
                    )
                )
            if str(event.get("kind", "")) == "mission_load":
                event_payload = dict(event.get("payload", {}) or {})
                manifest = dict(event_payload.get("manifest", {}) or {})
                load_id = str(manifest.get("load_id", "") or "")
                revision = _int_or_none(manifest.get("revision"))
                disposition = load_dispositions.get(
                    (int(invocation_id or 0), load_id, revision),
                    "delivery_unresolved",
                )
                load_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        _none_or_str(load_id),
                        revision,
                        disposition,
                        detail,
                    )
                )
                for raw_requirement in list(event_payload.get("safety_requirements", []) or []):
                    requirement = dict(raw_requirement or {})
                    activation_ns = _clock_ns(event.get("delivery_time"))
                    if disposition != "accepted" or activation_ns is None:
                        satisfied, source, assessment = (
                            None,
                            "load_not_accepted",
                            {"status": disposition},
                        )
                    else:
                        next_activation_ns = next(
                            (
                                value
                                for value in accepted_activation_times_ns
                                if value > activation_ns
                            ),
                            None,
                        )
                        start_s = activation_ns * 1.0e-9
                        end_s = None if next_activation_ns is None else next_activation_ns * 1.0e-9
                        sample_mask = np.asarray(t_s, dtype=float) >= start_s
                        if end_s is not None:
                            sample_mask &= np.asarray(t_s, dtype=float) < end_s
                        window_time_s = np.asarray(t_s, dtype=float)[sample_mask]
                        window_truth = {
                            key: np.asarray(values)[: len(sample_mask)][sample_mask[: len(np.asarray(values))]]
                            for key, values in truth_hist.items()
                        }
                        satisfied, source, assessment = _assess_safety_requirement(
                            str(object_id),
                            requirement,
                            t_s=window_time_s,
                            truth_hist=window_truth,
                        )
                        assessment = {
                            **assessment,
                            "activation_start_s": start_s,
                            "activation_end_s": end_s,
                        }
                    safety_rows.append(
                        (
                            str(object_id),
                            invocation_id,
                            _none_or_str(requirement.get("requirement_id")),
                            satisfied,
                            source,
                            json.dumps(
                                {**requirement, "assessment": assessment},
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    )

        for raw_output in list(evidence.get("outputs", []) or []):
            output = dict(raw_output or {})
            invocation_id = int(output.get("invocation_id", 0) or 0)
            for raw_command in list(output.get("commands", []) or []):
                command = dict(raw_command or {})
                packet = _packet_key(command.get("command_id"))
                validity = dict(command.get("validity", {}) or {})
                command_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        *packet,
                        _none_or_str(command.get("actuator_id")),
                        _clock_ns(command.get("issued_at")),
                        _clock_ns(validity.get("not_before")),
                        _clock_ns(validity.get("expires_at")),
                        _none_or_str(dict(command.get("payload", {}) or {}).get("schema")),
                        _debug_json(command, include=include_debug_detail),
                    )
                )
            for raw_telemetry in list(output.get("telemetry", []) or []):
                telemetry = dict(raw_telemetry or {})
                topic = _none_or_str(telemetry.get("topic"))
                generated_time_ns = _clock_ns(telemetry.get("generated_at"))
                telemetry_fields = [
                    item for item in list(telemetry.get("fields", []) or []) if isinstance(item, dict)
                ]
                fields = {
                    str(item.get("name")): item.get("value")
                    for item in telemetry_fields if item.get("name") is not None
                }
                diagnostic_rows.append(
                    (
                        str(object_id),
                        invocation_id,
                        topic,
                        generated_time_ns,
                        _debug_json(telemetry, include=include_debug_detail),
                    )
                )
                for field_index, field in enumerate(telemetry_fields):
                    if field.get("name") is None:
                        continue
                    kind, value_real, value_integer, value_text, value_json = _typed_review_value(
                        field.get("value")
                    )
                    diagnostic_field_rows.append(
                        (
                            str(object_id),
                            invocation_id,
                            topic,
                            generated_time_ns,
                            field_index,
                            str(field["name"]),
                            _none_or_str(field.get("unit", field.get("units"))),
                            kind,
                            value_real,
                            value_integer,
                            value_text,
                            value_json,
                        )
                    )
                if fields.get("goal_id") is not None or fields.get("goal_state") is not None:
                    objective_rows.append(
                        (
                            str(object_id),
                            invocation_id,
                            _none_or_str(fields.get("goal_id")),
                            _none_or_str(fields.get("goal_state")),
                            json.dumps(
                                {
                                    "goal_type": fields.get("goal_type"),
                                    "executive_phase": fields.get("executive_phase"),
                                    "selected_mode": fields.get("selected_mode"),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    )

        for raw_receipt in list(evidence.get("receipts", []) or []):
            receipt = dict(raw_receipt or {})
            packet = _packet_key(receipt.get("command_id"))
            receipt_rows.append(
                (
                    str(object_id),
                    *packet,
                    _clock_ns(receipt.get("received_at")),
                    _none_or_str(receipt.get("disposition")),
                    json.dumps(list(receipt.get("status_codes", []) or []), sort_keys=True),
                    _debug_json(receipt, include=include_debug_detail),
                )
            )

        for raw_realization in list(evidence.get("realizations", []) or []):
            realization = dict(raw_realization or {})
            source = realization.get("source_command_id")
            packet = _packet_key(source) if source else ("", "", None)
            interval_start_ns = _int_or_none(realization.get("interval_start_ns"))
            actuator_id = _none_or_str(realization.get("actuator_id"))
            realization_rows.append(
                (
                    str(object_id),
                    actuator_id,
                    interval_start_ns,
                    _int_or_none(realization.get("interval_end_ns")),
                    *packet,
                    _none_or_str(realization.get("demand_mode")),
                    _bool_int(realization.get("saturated")),
                    _float_or_none(realization.get("mass_flow_kg_s")),
                    *_vector3(realization.get("requested_force_n")),
                    *_vector3(realization.get("requested_torque_n_m")),
                    *_vector3(realization.get("realized_force_n")),
                    *_vector3(realization.get("realized_torque_n_m")),
                    _debug_json(realization, include=include_debug_detail),
                )
            )
            for device_index, raw_field in enumerate(list(realization.get("device_state", []) or [])):
                field = dict(raw_field or {})
                if field.get("name") is None:
                    continue
                kind, value_real, value_integer, value_text, value_json = _typed_review_value(field.get("value"))
                device_state_rows.append(
                    (
                        str(object_id),
                        actuator_id,
                        interval_start_ns,
                        device_index,
                        str(field["name"]),
                        _none_or_str(field.get("unit", field.get("units"))),
                        kind,
                        value_real,
                        value_integer,
                        value_text,
                        value_json,
                    )
                )

        for raw_snapshot in list(evidence.get("snapshots", []) or []):
            snapshot = dict(raw_snapshot or {})
            detail_json, detail_gzip = _snapshot_storage(
                snapshot, include_debug_detail=include_debug_detail
            )
            snapshot_rows.append(
                (
                    str(object_id),
                    _int_or_none(snapshot.get("invocation_id")),
                    _none_or_str(snapshot.get("stack_id")),
                    _none_or_str(snapshot.get("stack_version")),
                    _none_or_str(snapshot.get("state_hash_sha256")),
                    detail_json,
                    detail_gzip,
                )
            )

    conn.executemany("INSERT INTO fsw_invocations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", invocation_rows)
    conn.executemany("INSERT INTO fsw_input_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", input_rows)
    conn.executemany("INSERT INTO fsw_load_events VALUES (?, ?, ?, ?, ?, ?)", load_rows)
    conn.executemany("INSERT INTO fsw_objectives VALUES (?, ?, ?, ?, ?)", objective_rows)
    conn.executemany("INSERT INTO fsw_task_timing VALUES (?, ?, ?, ?, ?, ?, ?, ?)", task_rows)
    conn.executemany("INSERT INTO actuator_commands VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", command_rows)
    conn.executemany("INSERT INTO actuator_command_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?)", receipt_rows)
    conn.executemany(
        "INSERT INTO actuator_realization VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        realization_rows,
    )
    conn.executemany("INSERT INTO actuator_device_state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", device_state_rows)
    conn.executemany("INSERT INTO fsw_diagnostics VALUES (?, ?, ?, ?, ?)", diagnostic_rows)
    conn.executemany(
        "INSERT INTO fsw_diagnostic_fields VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", diagnostic_field_rows
    )
    conn.executemany("INSERT INTO safety_requirement_evidence VALUES (?, ?, ?, ?, ?, ?)", safety_rows)
    conn.executemany("INSERT INTO fsw_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", snapshot_rows)
    conn.executemany("INSERT INTO game_input_events VALUES (?, ?, ?, ?, ?, ?, ?)", game_input_rows)


def _assess_safety_requirement(
    object_id: str,
    requirement: dict[str, Any],
    *,
    t_s: np.ndarray,
    truth_hist: dict[str, np.ndarray],
) -> tuple[int | None, str, dict[str, Any]]:
    if str(requirement.get("evaluation", "")) != "quantitative":
        return None, "qualitative_review_required", {"status": "not_machine_assessable"}
    parameters = {
        str(item.get("name")): item.get("value")
        for item in list(requirement.get("parameters", []) or [])
        if isinstance(item, dict) and item.get("name") is not None
    }
    topics = tuple(str(value) for value in list(requirement.get("evidence_topics", []) or []))
    metric = str(parameters.get("metric", topics[0] if topics else "") or "").strip().lower()
    aliases = {
        "minimum_range_m": ("relative_range_m", ">=", parameters.get("minimum_range_m")),
        "maximum_range_m": ("relative_range_m", "<=", parameters.get("maximum_range_m")),
        "minimum_mass_kg": ("mass_kg", ">=", parameters.get("minimum_mass_kg")),
        "maximum_angular_rate_rad_s": (
            "angular_rate_rad_s",
            "<=",
            parameters.get("maximum_angular_rate_rad_s"),
        ),
        "maximum_speed_m_s": ("speed_m_s", "<=", parameters.get("maximum_speed_m_s")),
    }
    if metric in aliases:
        metric, default_operator, named_threshold = aliases[metric]
    else:
        default_operator, named_threshold = "<=", None
    operator = str(parameters.get("operator", default_operator) or default_operator)
    threshold_raw = parameters.get("threshold", named_threshold)
    if threshold_raw is None:
        return None, "quantitative_configuration_error", {
            "status": "missing_threshold",
            "metric": metric,
        }
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        return None, "quantitative_configuration_error", {
            "status": "invalid_threshold",
            "metric": metric,
        }
    own = truth_hist.get(object_id)
    if own is None:
        return None, "truth_evaluator", {"status": "object_truth_unavailable", "metric": metric}
    if metric == "mass_kg":
        values = own[:, 13]
    elif metric == "angular_rate_rad_s":
        values = np.linalg.norm(own[:, 10:13], axis=1)
    elif metric == "speed_m_s":
        values = np.linalg.norm(own[:, 3:6], axis=1) * 1.0e3
    elif metric == "relative_range_m":
        target_id = str(parameters.get("target_id", parameters.get("other_object_id", "")) or "")
        target = truth_hist.get(target_id)
        if target is None:
            return None, "truth_evaluator", {
                "status": "target_truth_unavailable",
                "metric": metric,
                "target_id": target_id,
            }
        size = min(len(own), len(target))
        values = np.linalg.norm(own[:size, :3] - target[:size, :3], axis=1) * 1.0e3
    else:
        return None, "quantitative_configuration_error", {
            "status": "unsupported_metric",
            "metric": metric,
        }
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, "truth_evaluator", {"status": "no_finite_samples", "metric": metric}
    comparisons = {
        "<=": values <= threshold,
        "<": values < threshold,
        ">=": values >= threshold,
        ">": values > threshold,
    }
    if operator not in comparisons:
        return None, "quantitative_configuration_error", {
            "status": "unsupported_operator",
            "operator": operator,
            "metric": metric,
        }
    compliance = comparisons[operator] | ~finite
    violation_indices = np.flatnonzero(~compliance)
    worst_index = int(np.nanargmax(values) if operator in {"<=", "<"} else np.nanargmin(values))
    return int(violation_indices.size == 0), "truth_evaluator", {
        "status": "satisfied" if violation_indices.size == 0 else "violated",
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "sample_count": int(np.count_nonzero(finite)),
        "violation_count": int(violation_indices.size),
        "first_violation_time_s": (
            None if violation_indices.size == 0 else float(t_s[int(violation_indices[0])])
        ),
        "worst_value": float(values[worst_index]),
        "worst_time_s": float(t_s[worst_index]),
    }


def _insert_game_evidence(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    observer_rows = []
    for raw_sample in list(payload.get("game_observer_samples", []) or []):
        sample = dict(raw_sample or {})
        observer_rows.append(
            (
                _none_or_str(sample.get("object_id")),
                _int_or_none(sample.get("time_ns")),
                _none_or_str(sample.get("observer_policy")),
                _bool_int(sample.get("truth_assisted")),
                json.dumps(dict(sample.get("detail", {}) or {}), sort_keys=True, separators=(",", ":")),
            )
        )
    scoring_rows = []
    for raw_event in list(payload.get("game_scoring_events", []) or []):
        event = dict(raw_event or {})
        scoring_rows.append(
            (
                _none_or_str(event.get("object_id")),
                _int_or_none(event.get("time_ns")),
                _none_or_str(event.get("scoring_policy")),
                _none_or_str(event.get("event_type")),
                json.dumps(dict(event.get("detail", {}) or {}), sort_keys=True, separators=(",", ":")),
            )
        )
    conn.executemany("INSERT INTO game_observer_samples VALUES (?, ?, ?, ?, ?)", observer_rows)
    conn.executemany("INSERT INTO game_scoring_events VALUES (?, ?, ?, ?, ?)", scoring_rows)


def _insert_ground_access(conn: sqlite3.Connection, *, t_s: np.ndarray, payload: dict[str, Any]) -> None:
    rows = []
    window_rows = []
    access_root = dict(payload.get("ground_station_access", {}) or {})
    for station_id, station_payload in access_root.items():
        targets = dict(dict(station_payload or {}).get("targets", {}) or {})
        for object_id, object_payload_raw in targets.items():
            object_payload = dict(object_payload_raw or {})
            access = list(object_payload.get("access", []) or [])
            los = list(object_payload.get("line_of_sight", []) or [])
            ranges = list(object_payload.get("range_km", []) or [])
            elevations = list(object_payload.get("elevation_deg", []) or [])
            reasons = list(object_payload.get("reason", []) or [])
            n = max(len(access), len(los), len(ranges), len(elevations), len(reasons))
            n = int(min(n, t_s.size))
            for i in range(n):
                rows.append(
                    (
                        i,
                        _finite_float(t_s[i]),
                        str(station_id),
                        str(object_id),
                        _bool_int(_list_get(access, i)),
                        _bool_int(_list_get(los, i)),
                        _float_or_none(_list_get(ranges, i)),
                        _float_or_none(_list_get(elevations, i)),
                        None if _list_get(reasons, i) is None else str(_list_get(reasons, i)),
                    )
                )
            access_flags = [bool(_list_get(access, i)) for i in range(n)]
            window_index = 0
            i = 0
            while i < n:
                if not access_flags[i]:
                    i += 1
                    continue
                start_i = i
                while i + 1 < n and access_flags[i + 1]:
                    i += 1
                end_i = i
                window_ranges = [_float_or_none(_list_get(ranges, j)) for j in range(start_i, end_i + 1)]
                window_elevations = [_float_or_none(_list_get(elevations, j)) for j in range(start_i, end_i + 1)]
                finite_ranges = [value for value in window_ranges if value is not None]
                finite_elevations = [value for value in window_elevations if value is not None]
                start_s = _finite_float(t_s[start_i])
                end_s = _finite_float(t_s[end_i])
                first_no_access_s = _finite_float(t_s[end_i + 1]) if end_i + 1 < n else None
                sample_span_s = None if start_s is None or end_s is None else end_s - start_s
                credited_end_s = first_no_access_s if first_no_access_s is not None else end_s
                credited_duration_s = (
                    None if start_s is None or credited_end_s is None else credited_end_s - start_s
                )
                window_rows.append(
                    (
                        str(station_id),
                        str(object_id),
                        window_index,
                        start_s,
                        end_s,
                        credited_duration_s,
                        int(start_i == 0),
                        int(end_i == n - 1),
                        min(finite_ranges) if finite_ranges else None,
                        max(finite_elevations) if finite_elevations else None,
                        (
                            "start_s/end_s are first/last retained access samples; duration_s is the "
                            "left-endpoint credited duration through first_no_access_sample_s when present"
                        ),
                        end_s,
                        first_no_access_s,
                        sample_span_s,
                        credited_duration_s,
                        "sampled_no_interpolation",
                    )
                )
                window_index += 1
                i += 1
    conn.executemany("INSERT INTO ground_access VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO ground_access_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", window_rows
    )


def _insert_orbital_analysis(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    root = dict(
        payload.get("_orbital_analysis_review", payload.get("orbital_analysis", {})) or {}
    )
    coverage_summary_rows = []
    coverage_sample_rows = []
    coverage_interval_rows = []
    coverage_transition_rows = []
    for raw in list(root.get("coverage", []) or []):
        item = dict(raw or {})
        analysis_id = str(item.get("analysis_id") or "")
        coverage_summary_rows.append((
            analysis_id, _none_or_str(item.get("source_object_id")), _none_or_str(item.get("state_provider_id")),
            _none_or_str(item.get("product_kind")), _none_or_str(item.get("attitude_source_kind")),
            _none_or_str(item.get("attitude_provider_id")), _none_or_str(item.get("refinement_source")),
            _none_or_str(item.get("input_evidence_sha256")),
            _none_or_str(item.get("semantic_sha256")), json.dumps(dict(item.get("summary", {}) or {}), sort_keys=True, separators=(",", ":")),
        ))
        for sample in list(item.get("samples", []) or []):
            row = dict(sample or {})
            coverage_sample_rows.append((analysis_id, _int_or_none(row.get("sample_index")), _float_or_none(row.get("time_s")),
                                         _int_or_none(row.get("covered_cell_count")), _float_or_none(row.get("instantaneous_covered_fraction"))))
        for interval in list(item.get("intervals", []) or []):
            row = dict(interval or {})
            coverage_interval_rows.append((
                analysis_id, _int_or_none(row.get("cell_index")), _int_or_none(row.get("interval_index")),
                _float_or_none(row.get("start_s")), _float_or_none(row.get("end_s")), _float_or_none(row.get("duration_s")),
                _bool_int(row.get("start_censored")), _bool_int(row.get("end_censored")),
                _none_or_str(row.get("acquisition_disposition")), _none_or_str(row.get("loss_disposition")),
                _none_or_str(row.get("acquisition_reason")), _none_or_str(row.get("loss_reason")),
            ))
        for transition in list(item.get("transitions", []) or []):
            row = dict(transition or {})
            coverage_transition_rows.append((
                analysis_id, _int_or_none(row.get("cell_index")), _none_or_str(row.get("transition_kind")),
                _float_or_none(row.get("time_s")), _float_or_none(row.get("bracket_start_s")),
                _float_or_none(row.get("bracket_end_s")), _none_or_str(row.get("disposition")),
                _int_or_none(row.get("iterations")), _none_or_str(row.get("reason_before")),
                _none_or_str(row.get("reason_after")),
            ))
    conn.executemany("INSERT INTO coverage_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", coverage_summary_rows)
    conn.executemany("INSERT INTO coverage_samples VALUES (?, ?, ?, ?, ?)", coverage_sample_rows)
    conn.executemany("INSERT INTO coverage_intervals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", coverage_interval_rows)
    conn.executemany("INSERT INTO coverage_transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", coverage_transition_rows)

    link_summary_rows = []
    link_sample_rows = []
    link_window_rows = []
    link_transition_rows = []
    for raw in list(root.get("directed_links", []) or []):
        item = dict(raw or {})
        analysis_id = str(item.get("analysis_id") or "")
        link_summary_rows.append((
            analysis_id, _none_or_str(item.get("link_id")), _none_or_str(item.get("tx_object_id")), _none_or_str(item.get("rx_object_id")),
            _none_or_str(item.get("tx_endpoint_kind")), _none_or_str(item.get("rx_endpoint_kind")),
            _none_or_str(item.get("tx_terminal_parent_frame")), _none_or_str(item.get("rx_terminal_parent_frame")),
            _none_or_str(item.get("tx_state_provider_id")), _none_or_str(item.get("rx_state_provider_id")),
            _none_or_str(item.get("tx_attitude_source_kind")), _none_or_str(item.get("tx_attitude_provider_id")),
            _none_or_str(item.get("rx_attitude_source_kind")), _none_or_str(item.get("rx_attitude_provider_id")),
            json.dumps(dict(item.get("refinement_source", {}) or {}), sort_keys=True, separators=(",", ":")),
            _none_or_str(item.get("input_evidence_sha256")),
            _none_or_str(item.get("semantic_sha256")), json.dumps(dict(item.get("summary", {}) or {}), sort_keys=True, separators=(",", ":")),
        ))
        for sample in list(item.get("samples", []) or []):
            row = dict(sample or {})
            link_sample_rows.append((analysis_id, _int_or_none(row.get("sample_index")), _float_or_none(row.get("time_s")),
                                     _float_or_none(row.get("range_km")), _float_or_none(row.get("margin_db")),
                                     _bool_int(row.get("available")), _none_or_str(row.get("primary_reason"))))
        for window in list(item.get("windows", []) or []):
            row = dict(window or {})
            link_window_rows.append((
                analysis_id, _int_or_none(row.get("interval_index")), _float_or_none(row.get("start_s")),
                _float_or_none(row.get("end_s")), _float_or_none(row.get("duration_s")),
                _bool_int(row.get("start_censored")), _bool_int(row.get("end_censored")),
                _none_or_str(row.get("acquisition_disposition")), _none_or_str(row.get("loss_disposition")),
                _float_or_none(row.get("minimum_margin_db")), _float_or_none(row.get("mean_margin_db")),
                _float_or_none(row.get("maximum_margin_db")), _float_or_none(row.get("minimum_range_km")),
                _float_or_none(row.get("estimated_delivered_data_bits")),
            ))
        for transition in list(item.get("transitions", []) or []):
            row = dict(transition or {})
            link_transition_rows.append((
                analysis_id, _none_or_str(row.get("transition_kind")), _float_or_none(row.get("time_s")),
                _float_or_none(row.get("bracket_start_s")), _float_or_none(row.get("bracket_end_s")),
                _none_or_str(row.get("disposition")), _int_or_none(row.get("iterations")),
                _none_or_str(row.get("reason_before")), _none_or_str(row.get("reason_after")),
            ))
    conn.executemany(
        "INSERT INTO link_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        link_summary_rows,
    )
    conn.executemany("INSERT INTO link_samples VALUES (?, ?, ?, ?, ?, ?, ?)", link_sample_rows)
    conn.executemany("INSERT INTO link_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", link_window_rows)
    conn.executemany("INSERT INTO link_transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", link_transition_rows)


def _insert_impulsive_maneuvers(conn: sqlite3.Connection, *, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload.get("impulsive_maneuvers", ()):
        event = dict(item)
        dv = list(event["delta_v_eci_m_s"])
        pre = list(event["pre_state_eci"])
        post = list(event["post_state_eci"])
        if len(dv) != 3 or len(pre) != 6 or len(post) != 6:
            raise ValueError("Malformed impulsive maneuver state evidence")
        rows.append((
            str(event["maneuver_id"]), str(event["object_id"]), int(event["sample_index"]),
            float(event["time_s"]), str(event["frame"]),
            *(float(value) for value in (*dv, *pre, *post)),
        ))
    conn.executemany(
        "INSERT INTO impulsive_maneuvers VALUES (" + ", ".join("?" for _ in range(20)) + ")",
        rows,
    )


def _insert_events(
    conn: sqlite3.Connection,
    *,
    t_s: np.ndarray,
    summary: dict[str, Any],
    thrust_hist: dict[str, np.ndarray],
    impulsive_maneuvers: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    collision_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> None:
    rows = []
    for index, impact in enumerate(collision_events):
        event_time = float(impact["time_s"])
        object_ids = list(impact["object_ids"])
        rows.append((
            f"spherical_collision:{index}",
            event_time,
            _sample_index_for_time(t_s, event_time),
            ",".join(object_ids),
            "spherical_collision",
            "info",
            f"Elastic spherical collision between {object_ids[0]} and {object_ids[1]}; "
            f"closing speed {float(impact['closing_speed_m_s']):.9g} m/s; "
            f"pre/post ECI velocities (km/s): {impact['pre_velocity_eci_km_s']} -> "
            f"{impact['post_velocity_eci_km_s']}",
            "collision_dynamics",
        ))
    if bool(summary.get("terminated_early", False)):
        event_time = _float_or_none(summary.get("termination_time_s"))
        rows.append(
            (
                "termination",
                event_time,
                _sample_index_for_time(t_s, event_time),
                _none_or_str(summary.get("termination_object_id")),
                "termination",
                "warning",
                _none_or_str(summary.get("termination_reason")) or "simulation terminated early",
                "summary",
            )
        )
    mission_recovery = dict(summary.get("mission_recovery", {}) or {})
    planner = dict(mission_recovery.get("planner", {}) or {})
    rejections = dict(planner.get("candidate_rejections", {}) or {})
    rejection_counts = {
        str(reason): int(count)
        for reason, count in dict(rejections.get("by_reason", {}) or {}).items()
        if int(count) > 0
    }
    rejection_total = int(sum(rejection_counts.values()))
    if rejection_total:
        event_time = _float_or_none(mission_recovery.get("assessment_time_s"))
        object_id = _none_or_str(mission_recovery.get("object_id"))
        reason_text = ", ".join(f"{reason}={count}" for reason, count in sorted(rejection_counts.items()))
        rows.append(
            (
                f"mission_recovery_candidate_rejection:{object_id or 'unknown'}",
                event_time,
                _sample_index_for_time(t_s, event_time),
                object_id,
                "mission_recovery_candidate_rejection",
                "warning",
                (
                    f"Mission-recovery planner rejected {rejection_total} sampled candidate grid points "
                    f"and continued without aborting the planner ({reason_text})."
                ),
                "mission_recovery_planner",
            )
        )
    for impulse in impulsive_maneuvers:
        dv = np.asarray(impulse["delta_v_eci_m_s"], dtype=float)
        rows.append((
            f"impulsive_maneuver:{impulse['maneuver_id']}",
            float(impulse["time_s"]), int(impulse["sample_index"]),
            str(impulse["object_id"]), "impulsive_maneuver", "info",
            f"{impulse['maneuver_id']} applied {float(np.linalg.norm(dv)):.9g} m/s ECI impulse",
            "impulsive_maneuvers",
        ))
    for object_id, hist in thrust_hist.items():
        if hist.shape[1] < 3:
            continue
        n = int(min(t_s.size, hist.shape[0]))
        active = np.linalg.norm(hist[:n, :3], axis=1) > 0.0
        previous = False
        for i, current in enumerate(active):
            if bool(current) and not previous:
                start_index = max(i - 1, 0)
                rows.append(
                    (
                        f"burn_start:{object_id}:{i}",
                        _finite_float(t_s[start_index]),
                        start_index,
                        object_id,
                        "burn_start",
                        "info",
                        f"{object_id} burn interval started",
                        "review_store",
                    )
                )
            if previous and not bool(current):
                end_index = max(i - 1, 0)
                rows.append(
                    (
                        f"burn_end:{object_id}:{i}",
                        _finite_float(t_s[end_index]),
                        end_index,
                        object_id,
                        "burn_end",
                        "info",
                        f"{object_id} burn interval ended",
                        "review_store",
                    )
                )
            previous = bool(current)
        if previous and n > 0:
            rows.append(
                (
                    f"burn_end:{object_id}:{n - 1}:final",
                    _finite_float(t_s[n - 1]),
                    n - 1,
                    object_id,
                    "burn_end",
                    "info",
                    f"{object_id} burn interval ended at final sample",
                    "review_store",
                )
            )
    consistency = dict(summary.get("knowledge_consistency_by_observer", {}) or {})
    for observer_id, targets_raw in sorted(consistency.items()):
        for target_id, evidence_raw in sorted(dict(targets_raw or {}).items()):
            evidence = dict(evidence_raw or {})
            confirmed_count = int(evidence.get("maneuver_confirmed_event_count", 0) or 0)
            confirmed_time = _float_or_none(evidence.get("maneuver_first_confirmed_t_s"))
            if confirmed_count <= 0 or confirmed_time is None:
                continue
            sample_index = _sample_index_for_time(t_s, confirmed_time)
            rows.append(
                (
                    f"maneuver_detection_confirmed:{observer_id}:{target_id}:{sample_index}",
                    confirmed_time,
                    sample_index,
                    str(target_id),
                    "maneuver_detection_confirmed",
                    "warning",
                    (
                        f"{observer_id} confirmed a maneuver by {target_id}; "
                        f"confirmed_event_count={confirmed_count}, max_nis={evidence.get('maneuver_max_nis')}"
                    ),
                    "knowledge_maneuver_detector",
                )
            )
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_mission_recovery(conn: sqlite3.Connection, *, summary: dict[str, Any]) -> None:
    recovery = dict(summary.get("mission_recovery", {}) or {})
    if not recovery:
        return
    estimate = dict(recovery.get("recovery_estimate", {}) or {})
    conn.execute(
        """
        INSERT INTO mission_recovery_summary VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            _none_or_str(recovery.get("object_id")),
            _none_or_str(recovery.get("goal")),
            _none_or_str(estimate.get("method")),
            _float_or_none(recovery.get("assessment_time_s")),
            _int_or_none(recovery.get("assessment_sample_index")),
            _bool_int(estimate.get("available")),
            _float_or_none(estimate.get("recovery_delta_v_m_s")),
            _float_or_none(estimate.get("recovery_time_s")),
            _none_or_str(estimate.get("recovery_time_basis")),
            _float_or_none(estimate.get("propellant_kg")),
            _float_or_none(estimate.get("propellant_fraction")),
            _float_or_none(estimate.get("disturbance_delta_v_m_s")),
            _none_or_str(estimate.get("disturbance_apsis")),
            _bool_int(estimate.get("slot_recovery_found")),
            _int_or_none(estimate.get("slot_recovery_orbits")),
            _float_or_none(estimate.get("slot_recovery_time_s")),
            _float_or_none(estimate.get("slot_recovery_phase_error_deg")),
            _int_or_none(estimate.get("best_slot_orbits")),
            _float_or_none(estimate.get("best_slot_time_s")),
            _float_or_none(estimate.get("best_slot_phase_error_deg")),
            _float_or_none(estimate.get("local_orbit_shape_delta_v_m_s")),
            _float_or_none(estimate.get("local_orbit_shape_position_error_km")),
            json.dumps(list(estimate.get("notes", []) or []), sort_keys=True),
        ),
    )
    rows = []
    object_id = _none_or_str(recovery.get("object_id"))
    for label in ("initial", "target", "final"):
        elements = dict(recovery.get(f"{label}_elements", {}) or {})
        if elements:
            rows.append(
                (
                    object_id,
                    label,
                    _float_or_none(elements.get("a_km")),
                    _float_or_none(elements.get("ecc")),
                    _float_or_none(elements.get("inc_deg")),
                    _float_or_none(elements.get("raan_deg")),
                    _float_or_none(elements.get("argp_deg")),
                    _float_or_none(elements.get("true_anomaly_deg")),
                )
            )
    conn.executemany("INSERT INTO mission_recovery_elements VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    _insert_mission_recovery_planner(conn, recovery=recovery)


def _insert_mission_recovery_planner(conn: sqlite3.Connection, *, recovery: dict[str, Any]) -> None:
    planner = dict(recovery.get("planner", {}) or {})
    if not planner:
        return
    object_id = _none_or_str(recovery.get("object_id"))
    recommended = dict(planner.get("recommended", {}) or {})
    recommended_by_candidate: dict[str, list[str]] = {}
    for mode, candidate_id in recommended.items():
        if candidate_id:
            recommended_by_candidate.setdefault(str(candidate_id), []).append(str(mode))
    candidate_rows = []
    burn_rows = []
    element_rows = []
    for candidate_raw in list(planner.get("candidates", []) or []):
        candidate = dict(candidate_raw or {})
        candidate_id = _none_or_str(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        candidate_rows.append(
            (
                candidate_id,
                object_id,
                _none_or_str(candidate.get("goal", recovery.get("goal"))),
                _none_or_str(candidate.get("source")),
                _none_or_str(candidate.get("source_family")),
                _none_or_str(candidate.get("target_basis")),
                _none_or_str(candidate.get("description")),
                _float_or_none(candidate.get("planned_delta_v_m_s")),
                _float_or_none(candidate.get("simulated_delta_v_m_s")),
                _float_or_none(candidate.get("planned_time_s")),
                _float_or_none(candidate.get("simulated_recovery_time_s")),
                _float_or_none(candidate.get("propellant_kg")),
                _float_or_none(candidate.get("propellant_fraction")),
                _bool_int(candidate.get("feasible")),
                _bool_int(candidate.get("verified")),
                _bool_int(candidate.get("within_tolerances")),
                _float_or_none(candidate.get("score")),
                json.dumps(recommended_by_candidate.get(candidate_id, []), sort_keys=True),
                _none_or_str(candidate.get("transfer_type")),
                _float_or_none(candidate.get("departure_wait_s")),
                _float_or_none(candidate.get("time_of_flight_s")),
                _float_or_none(candidate.get("arrival_time_s")),
                _float_or_none(candidate.get("target_phase_deg")),
                _bool_int(candidate.get("lambert_short_way")),
                _int_or_none(candidate.get("lambert_revolutions")),
                _int_or_none(candidate.get("solver_iterations")),
                _float_or_none(candidate.get("solver_residual_s")),
                _float_or_none(candidate.get("position_residual_km")),
                _float_or_none(candidate.get("velocity_residual_m_s")),
                json.dumps(list(candidate.get("notes", []) or []), sort_keys=True),
            )
        )
        for burn_raw in list(candidate.get("burn_sequence", []) or []):
            burn = dict(burn_raw or {})
            burn_rows.append(
                (
                    candidate_id,
                    _int_or_none(burn.get("burn_index")),
                    _float_or_none(burn.get("start_time_s")),
                    _float_or_none(burn.get("duration_s")),
                    _none_or_str(burn.get("frame")),
                    _none_or_str(burn.get("axis")),
                    _float_or_none(burn.get("delta_v_m_s")),
                    json.dumps(list(burn.get("delta_v_eci_m_s", []) or []), sort_keys=True),
                )
            )
        elements = dict(candidate.get("expected_final_elements", {}) or {})
        if elements:
            element_rows.append(
                (
                    candidate_id,
                    object_id,
                    _float_or_none(elements.get("a_km")),
                    _float_or_none(elements.get("ecc")),
                    _float_or_none(elements.get("inc_deg")),
                    _float_or_none(elements.get("raan_deg")),
                    _float_or_none(elements.get("argp_deg")),
                    _float_or_none(elements.get("true_anomaly_deg")),
                    json.dumps(dict(candidate.get("expected_element_errors", {}) or {}), sort_keys=True),
                )
            )
    conn.executemany(
        "INSERT INTO mission_recovery_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        candidate_rows,
    )
    conn.executemany("INSERT INTO mission_recovery_burns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", burn_rows)
    conn.executemany(
        "INSERT INTO mission_recovery_candidate_elements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        element_rows,
    )


def _insert_metrics(conn: sqlite3.Connection, *, summary: dict[str, Any]) -> None:
    rows = []
    pair = [str(item) for item in list(summary.get("primary_object_pair", []) or [])]
    deputy_id = pair[0] if len(pair) == 2 else None
    chief_id = pair[1] if len(pair) == 2 else None
    rel_summary = dict(summary.get("relative_range_summary", {}) or {})
    for name, units in (
        ("initial_range_km", "km"),
        ("final_range_km", "km"),
        ("closest_approach_km", "km"),
        ("closest_approach_time_s", "s"),
    ):
        if name in rel_summary:
            rows.append((name, name, None, deputy_id, chief_id, _float_or_none(rel_summary.get(name)), units, "summary"))
    thrust_stats = dict(summary.get("thrust_stats", {}) or {})
    for object_id, stats_raw in thrust_stats.items():
        stats = dict(stats_raw or {})
        for name, units in (
            ("burn_samples", "samples"),
            ("max_accel_km_s2", "km/s^2"),
            ("total_dv_m_s", "m/s"),
        ):
            if name in stats:
                metric_id = f"{object_id}:{name}"
                rows.append((metric_id, name, str(object_id), None, None, _float_or_none(stats.get(name)), units, "summary"))
    # Recovery counts and acquisition/decision epochs are distinct from maneuver
    # confirmations. Only opt-in tracks emit these keys; legacy runs are unchanged.
    for observer_id, targets in dict(summary.get("knowledge_consistency_by_observer", {}) or {}).items():
        for target_id, evidence in dict(targets or {}).items():
            evidence = dict(evidence or {})
            for name, units in (
                ("recovery_hold_count", "samples"),
                ("recovery_reinitialization_count", "events"),
                ("recovery_transient_count", "events"),
                ("last_recovery_event_t_s", "s"),
                ("last_recovery_decision_t_s", "s"),
            ):
                if name in evidence:
                    metric_id = f"{observer_id}:{target_id}:{name}"
                    rows.append((metric_id, name, str(observer_id), None, str(target_id),
                                 _float_or_none(evidence[name]), units, "innovation_recovery"))
    recovery = dict(summary.get("mission_recovery", {}) or {})
    estimate = dict(recovery.get("recovery_estimate", {}) or {})
    object_id = str(recovery.get("object_id", "") or "")
    for name, units in (
        ("recovery_delta_v_m_s", "m/s"),
        ("recovery_time_s", "s"),
        ("propellant_kg", "kg"),
        ("local_orbit_shape_position_error_km", "km"),
    ):
        if name in estimate:
            metric_id = f"{object_id}:mission_recovery:{name}" if object_id else f"mission_recovery:{name}"
            rows.append((metric_id, name, object_id or None, None, None, _float_or_none(estimate.get(name)), units, "summary"))
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def _insert_artifacts(
    conn: sqlite3.Connection,
    *,
    artifacts: dict[str, Any],
    outdir: Path,
    generated_utc: str,
) -> None:
    rows = []
    for artifact_type, artifact_id, path in _iter_artifact_paths(artifacts):
        rows.append(
            (
                artifact_id,
                artifact_type,
                _relative_artifact_path(path, outdir),
                artifact_id.replace("_", " ").title(),
                "single_run_artifacts",
                generated_utc,
            )
        )
    conn.executemany("INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?, ?)", rows)


def _write_schema_json(path: Path, *, generated_utc: str, db_path: Path | None = None) -> None:
    schema = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "format": "sqlite",
        "database": "run.sqlite",
        "compatibility": {
            "policy": REVIEW_SCHEMA_COMPATIBILITY_POLICY,
            "stable_tables": list(REVIEW_SCHEMA_STABLE_TABLES),
            "breaking_change_requires_schema_version_bump": True,
            "reader_guidance": (
                "Prefer table/column discovery through this schema and ReviewWorkspace.schema(); "
                "new tables or nullable columns may be added before 1.0."
            ),
        },
        "tables": {
            "run_metadata": {"description": "One row describing the run and review schema."},
            "objects": {"description": "Active simulation objects."},
            "frame_provenance": {"description": "Scenario-level frame, EOP, and time-scale provenance."},
            "object_initialization": {
                "description": "Per-object initial-state provenance, including TLE recovery handoffs."
            },
            "object_propagation": {
                "description": "Resolved per-object propagator family/name, OGP regime, frames, period, and TLE-age provenance."
            },
            "object_state_frame": {"description": "Frame label for each object's state rows, e.g. eci or teme."},
            "time_samples": {"description": "Retained sample times."},
            "object_state": {"description": "Truth state histories by object; join object_state_frame for frame labels."},
            "coverage_summary": {"description": "One source-bound summary row per whole-Earth coverage analysis."},
            "coverage_samples": {"description": "Time-indexed whole-Earth covered-cell counts and fractions."},
            "coverage_intervals": {"description": "Sample-bounded, provider-refined, or censored cell coverage intervals."},
            "coverage_transitions": {"description": "Coverage acquisition/loss brackets and provider-refinement evidence."},
            "link_summary": {"description": "One provenance-bound summary row per directed link analysis."},
            "link_samples": {"description": "Time-indexed directed-link range, margin, availability, and reason."},
            "link_windows": {"description": "Directed-link availability windows with margin and volume evidence."},
            "link_transitions": {"description": "Acquisition/loss brackets and provider-refinement dispositions."},
            "relative_state": {
                "description": "RIC relative state for configured/default review object pairs."
            },
            "thrust": {"description": "Applied acceleration histories by object."},
            "controller_decisions": {
                "description": "Compact always-on controller and command decision records by object."
            },
            "mission_modes": {
                "description": "Mission strategy, execution, phase, and executive mode for each decision."
            },
            "mission_transitions": {
                "description": "Mission-executive mode transitions and their trigger evidence."
            },
            "command_gates": {
                "description": "Requested/applied burn state and alignment, fuel, actuator, and deadline gates."
            },
            "fsw_invocations": {
                "description": "Complete-stack invocations with exact input and command membership."
            },
            "fsw_input_events": {
                "description": "Typed input events delivered across the satellite FSW boundary."
            },
            "fsw_load_events": {"description": "Mission and stack load lifecycle evidence."},
            "fsw_objectives": {"description": "Onboard objective declarations and state transitions."},
            "fsw_task_timing": {"description": "Requested and modeled onboard task releases and budgets."},
            "actuator_commands": {
                "description": "Typed actuator commands linked to their source stack invocation."
            },
            "actuator_command_receipts": {
                "description": "Command-bus acceptance or rejection evidence linked by command identity."
            },
            "actuator_realization": {
                "description": "Normalized force, torque, mass-flow, saturation, and command-linked realization evidence."
            },
            "actuator_device_state": {
                "description": "Typed per-device state fields linked to actuator realization intervals."
            },
            "fsw_diagnostics": {
                "description": "Diagnostic envelope identities; full detail_json is retained only at full detail."
            },
            "fsw_diagnostic_fields": {
                "description": "Typed qualification-safe diagnostic telemetry fields without duplicated envelopes."
            },
            "safety_requirement_evidence": {
                "description": "Truth-derived post-run assessment of configured safety requirements."
            },
            "fsw_snapshots": {"description": "Checkpoint identities and hashes for complete FSW stacks."},
            "game_input_events": {"description": "Typed game pilot events linked to stack invocations."},
            "game_observer_samples": {
                "description": "Presentation observer samples, kept outside the onboard input boundary."
            },
            "game_scoring_events": {
                "description": "Truth-derived game scoring events, kept outside the onboard input boundary."
            },
            "attitude_error": {
                "description": "Attitude error histories; desired/actual quaternion fields are populated when retained, otherwise controller telemetry supplies the error angle."
            },
            "ground_access": {"description": "Ground station access histories."},
            "ground_access_windows": {
                "description": (
                    "Normalized sampled access windows with explicit retained-sample bounds, first no-access "
                    "sample, credited duration, refinement status, and run-boundary censoring flags."
                )
            },
            "object_orbital_elements": {
                "description": "Derived ECI radius, speed, and classical elements with circular/equatorial conditioning flags."
            },
            "events": {"description": "Termination and review-derived event rows."},
            "metrics": {"description": "Scalar summary and review metrics."},
            "mission_recovery_summary": {
                "description": "Configured mission-recovery estimate from initial and assessment-state orbit comparison."
            },
            "mission_recovery_elements": {
                "description": "Initial and assessment classical orbital elements used by mission recovery."
            },
            "mission_recovery_candidates": {
                "description": "Planner candidate trade-space rows for configured mission recovery and Orbit Transfer Planner analyses."
            },
            "mission_recovery_burns": {
                "description": "Burn sequence rows for mission-recovery planner candidates."
            },
            "mission_recovery_candidate_elements": {
                "description": "Expected candidate final elements and element-error JSON."
            },
            "artifacts": {"description": "Known artifacts in the output folder."},
        },
    }
    if db_path is not None:
        conn = sqlite3.connect(db_path)
        try:
            table_names = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            for table_name in table_names:
                entry = dict(schema["tables"].get(table_name, {}) or {})
                entry.setdefault("description", f"Review evidence table: {table_name}.")
                entry["columns"] = [
                    {
                        "name": str(row[1]),
                        "type": str(row[2] or ""),
                        "notnull": bool(row[3]),
                        "primary_key": bool(row[5]),
                        **_review_column_metadata(str(row[1])),
                    }
                    for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                ]
                schema["tables"][table_name] = entry
        finally:
            conn.close()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _review_column_metadata(column_name: str) -> dict[str, Any]:
    """Describe every discoverable column and expose units where applicable."""

    name = str(column_name)
    unit: str | None = None
    for suffix, candidate in (
        ("_n_m", "N m"),
        ("_kg_s", "kg/s"),
        ("_km_s", "km/s"),
        ("_rad_s", "rad/s"),
        ("_db_hz", "dB-Hz"),
        ("_dbw", "dBW"),
        ("_dbi", "dBi"),
        ("_deg", "deg"),
        ("_rad", "rad"),
        ("_km", "km"),
        ("_wh", "Wh"),
        ("_hz", "Hz"),
        ("_bytes", "byte"),
        ("_ns", "ns"),
        ("_n", "N"),
        ("_s", "s"),
        ("_fraction", "1"),
    ):
        if name.endswith(suffix):
            unit = candidate
            break
    if name.endswith("_count") or name in {"samples", "sample_index", "interval_index"}:
        unit = "count"
    if name in {
        "available",
        "access",
        "line_of_sight",
        "start_censored",
        "end_censored",
    }:
        unit = "boolean"
    return {
        "unit": unit,
        "description": name.replace("_", " ").strip().capitalize() + ".",
    }


def _iter_artifact_paths(artifacts: dict[str, Any], prefix: str = ""):
    for key, value in dict(artifacts or {}).items():
        path_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _iter_artifact_paths(value, path_key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                child_key = f"{path_key}[{index}]"
                if isinstance(child, dict):
                    yield from _iter_artifact_paths(child, child_key)
                elif isinstance(child, str) and child:
                    yield prefix or "artifact", child_key.replace(".", ":"), Path(child)
        elif isinstance(value, str) and value:
            yield prefix or "artifact", path_key.replace(".", ":"), Path(value)


def _relative_artifact_path(path: Path, outdir: Path) -> str:
    try:
        return path.resolve().relative_to(outdir.resolve()).as_posix()
    except Exception:
        return str(path)


def _as_2d_float(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _state_value(row: np.ndarray, idx: int) -> float | None:
    if idx >= row.size:
        return None
    return _finite_float(row[idx])


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _float_or_none(value: Any) -> float | None:
    return _finite_float(value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))


def _none_or_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _list_get(values: list[Any], idx: int) -> Any:
    return values[idx] if idx < len(values) else None


def _sample_index_for_time(t_s: np.ndarray, time_s: float | None) -> int | None:
    if time_s is None or t_s.size <= 0:
        return None
    return int(np.argmin(np.abs(t_s - float(time_s))))


def _utc_stamp() -> str:
    return os.environ.get("OEL_GENERATED_UTC", "").strip()
