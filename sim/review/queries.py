from __future__ import annotations

import re
from dataclasses import dataclass

SAVED_QUERY_MATURITY_LEVELS = frozenset({"supported", "prototype", "experimental"})


@dataclass(frozen=True)
class SavedReviewQuery:
    name: str
    description: str
    sql: str
    source_tables: tuple[str, ...] = ()
    maturity: str = "supported"
    allow_empty: bool = False
    max_vm_steps: int = 250_000

    def __post_init__(self) -> None:
        if self.maturity not in SAVED_QUERY_MATURITY_LEVELS:
            allowed = ", ".join(sorted(SAVED_QUERY_MATURITY_LEVELS))
            raise ValueError(f"Unknown saved review query maturity {self.maturity!r}; expected one of: {allowed}")
        if not self.sql.lstrip().upper().startswith(("SELECT", "WITH")):
            raise ValueError(f"Saved review query {self.name!r} must be read-only SELECT/WITH SQL.")
        if int(self.max_vm_steps) <= 0:
            raise ValueError(f"Saved review query {self.name!r} must have a positive VM step budget.")
        if not self.source_tables:
            object.__setattr__(self, "source_tables", _infer_source_tables(self.sql))


def _infer_source_tables(sql: str) -> tuple[str, ...]:
    cte_names = {
        match.group(1)
        for match in re.finditer(r"(?:\bWITH|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql, flags=re.IGNORECASE)
    }
    names = {
        match.group(1)
        for match in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE)
        if match.group(1).lower() not in {"select"} and match.group(1) not in cte_names
    }
    return tuple(sorted(names))


SAVED_REVIEW_QUERIES: dict[str, SavedReviewQuery] = {
    "run_metadata": SavedReviewQuery(
        name="run_metadata",
        description=(
            "Run identity, timing, engine/review versions, generated time, config digest, and source reference."
        ),
        sql="SELECT * FROM run_metadata",
    ),
    "objects": SavedReviewQuery(
        name="objects",
        description="Active object inventory.",
        sql="SELECT object_id, object_type, role, mass_initial_kg FROM objects ORDER BY object_id",
    ),
    "artifacts": SavedReviewQuery(
        name="artifacts",
        description="Known output artifact inventory.",
        sql="SELECT artifact_type, artifact_id, path FROM artifacts ORDER BY artifact_type, artifact_id",
    ),
    "passive_final_state": SavedReviewQuery(
        name="passive_final_state",
        description="Compatibility alias for final ECI state; the run may be passive or controlled.",
        sql=(
            "WITH bounds AS (SELECT object_id, MAX(sample_index) AS last_i FROM object_state "
            "GROUP BY object_id) SELECT s.object_id, s.time_s, s.pos_x_eci_km, s.pos_y_eci_km, "
            "s.pos_z_eci_km, s.vel_x_eci_km_s, s.vel_y_eci_km_s, s.vel_z_eci_km_s "
            "FROM object_state s JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index = b.last_i ORDER BY s.object_id"
        ),
    ),
    "object_final_state": SavedReviewQuery(
        name="object_final_state",
        description="Final ECI state with derived radius and speed for every object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MAX(sample_index) AS last_i FROM object_state "
            "GROUP BY object_id) SELECT s.object_id, s.time_s, s.pos_x_eci_km, s.pos_y_eci_km, "
            "s.pos_z_eci_km, s.vel_x_eci_km_s, s.vel_y_eci_km_s, s.vel_z_eci_km_s, "
            "e.radius_km, e.speed_km_s FROM object_state s "
            "JOIN bounds b ON s.object_id = b.object_id AND s.sample_index = b.last_i "
            "LEFT JOIN object_orbital_elements e USING (sample_index, time_s, object_id) "
            "ORDER BY s.object_id"
        ),
    ),
    "object_eci_radius_extrema": SavedReviewQuery(
        name="object_eci_radius_extrema",
        description="Minimum and maximum sampled ECI radius and speed by object.",
        sql=(
            "SELECT object_id, COUNT(*) AS samples, MIN(radius_km) AS minimum_radius_km, "
            "MAX(radius_km) AS maximum_radius_km, MIN(speed_km_s) AS minimum_speed_km_s, "
            "MAX(speed_km_s) AS maximum_speed_km_s FROM object_orbital_elements "
            "GROUP BY object_id ORDER BY object_id"
        ),
    ),
    "object_orbital_elements_first_last": SavedReviewQuery(
        name="object_orbital_elements_first_last",
        description="First/final derived classical elements with conditioning flags.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) first_i, MAX(sample_index) last_i "
            "FROM object_orbital_elements GROUP BY object_id) SELECT e.* FROM object_orbital_elements e "
            "JOIN bounds b ON e.object_id=b.object_id AND e.sample_index IN (b.first_i,b.last_i) "
            "ORDER BY e.object_id,e.sample_index"
        ),
    ),
    "ogp_propagation_contract": SavedReviewQuery(
        name="ogp_propagation_contract",
        description="Per-object OGP model, native/output frame, canonical state frame, and TLE-age provenance.",
        sql=(
            "SELECT p.object_id, p.propagation_method, p.propagator_family, p.propagator_name, "
            "p.general_model, p.ogp_regime, p.orbital_period_min, p.native_frame, "
            "p.output_frame, p.state_history_frame, p.frame_transform, f.state_frame, p.tle_epoch_jd_utc, "
            "p.tle_age_start_days, p.tle_age_end_days, p.max_tle_age_days_warning, p.tle_age_warning "
            "FROM object_propagation p "
            "LEFT JOIN object_state_frame f USING (object_id) ORDER BY p.object_id"
        ),
    ),
    "object_propagation_contract": SavedReviewQuery(
        name="object_propagation_contract",
        description=(
            "Branch-aware per-object initialization and continuous-propagation provenance, including "
            "TLE-to-ONP handoffs as well as continuous OGP runs."
        ),
        sql=(
            "SELECT o.object_id, i.source AS initialization_source, i.initialization_model, "
            "i.initialization_propagator_family, i.initialization_propagator_name, "
            "i.handoff_propagation_method, p.propagation_method, p.propagator_family, "
            "p.propagator_name, p.general_model, p.ogp_regime, p.orbital_period_min, "
            "COALESCE(p.native_frame, i.native_frame) AS native_frame, "
            "COALESCE(p.output_frame, i.output_frame) AS output_frame, p.state_history_frame, "
            "COALESCE(p.frame_transform, i.frame_transform) AS frame_transform, f.state_frame, "
            "COALESCE(p.tle_epoch_jd_utc, i.tle_epoch_jd_utc) AS tle_epoch_jd_utc, "
            "i.tle_age_initialization_days, p.tle_age_start_days, p.tle_age_end_days, "
            "p.max_tle_age_days_warning, p.tle_age_warning FROM objects o "
            "LEFT JOIN object_initialization i USING(object_id) "
            "LEFT JOIN object_propagation p USING(object_id) "
            "LEFT JOIN object_state_frame f USING(object_id) ORDER BY o.object_id"
        ),
    ),
    "object_state_first_last": SavedReviewQuery(
        name="object_state_first_last",
        description="First and final ECI position and velocity samples for each object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.sample_index, s.time_s, s.pos_x_eci_km, "
            "s.pos_y_eci_km, s.pos_z_eci_km, s.vel_x_eci_km_s, "
            "s.vel_y_eci_km_s, s.vel_z_eci_km_s FROM object_state s "
            "JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) "
            "ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "rendezvous_metrics": SavedReviewQuery(
        name="rendezvous_metrics",
        description="Initial range, final range, closest approach, and closest approach time.",
        sql=(
            "SELECT metric_name, value, units, deputy_id, chief_id FROM metrics "
            "WHERE metric_name IN ('initial_range_km', 'final_range_km', "
            "'closest_approach_km', 'closest_approach_time_s') ORDER BY metric_name"
        ),
    ),
    "rendezvous_closest_approach": SavedReviewQuery(
        name="rendezvous_closest_approach",
        description="Closest relative-state sample by range.",
        sql=(
            "SELECT time_s, deputy_id, chief_id, range_km, range_rate_km_s "
            "FROM relative_state WHERE range_km IS NOT NULL ORDER BY range_km ASC LIMIT 1"
        ),
    ),
    "relative_final_state": SavedReviewQuery(
        name="relative_final_state",
        description="Final relative-state sample.",
        sql=(
            "WITH bounds AS (SELECT deputy_id, chief_id, MAX(sample_index) AS last_i "
            "FROM relative_state GROUP BY deputy_id, chief_id) SELECT r.time_s, r.deputy_id, "
            "r.chief_id, r.r_radial_km, r.i_intrack_km, r.c_crosstrack_km, r.range_km, "
            "r.range_rate_km_s FROM relative_state r JOIN bounds b "
            "ON r.deputy_id = b.deputy_id AND r.chief_id = b.chief_id "
            "AND r.sample_index = b.last_i ORDER BY r.deputy_id, r.chief_id"
        ),
    ),
    "burn_activity": SavedReviewQuery(
        name="burn_activity",
        description="Applied acceleration and active burn sample counts by object.",
        sql=(
            "SELECT object_id, COUNT(*) AS samples, SUM(burn_active) AS active_samples, "
            "MAX(accel_norm_km_s2) AS max_accel_km_s2 FROM thrust GROUP BY object_id "
            "ORDER BY object_id"
        ),
        max_vm_steps=500_000,
    ),
    "burn_command_summary": SavedReviewQuery(
        name="burn_command_summary",
        description=(
            "Original scheduled-burn request, controller command, applied acceleration, saturation, and "
            "realized delta-v evidence by object. "
            "Applied thrust rows are retained samples, so active-sample counts are not continuous burn duration."
        ),
        sql=(
            "WITH decisions AS (SELECT object_id, COUNT(*) decision_samples, SUM(burn_requested) requested_samples, "
            "SUM(burn_applied) applied_decision_samples, SUM(saturated) saturated_samples, "
            "MAX(requested_accel_norm_km_s2) max_requested_accel_km_s2, "
            "MAX(applied_accel_norm_km_s2) max_applied_accel_km_s2 FROM controller_decisions GROUP BY object_id), "
            "realized AS (SELECT object_id, SUM(burn_active) active_thrust_samples, "
            "MAX(accel_norm_km_s2) max_realized_accel_km_s2 FROM thrust GROUP BY object_id), "
            "diagnostics AS (SELECT object_id, "
            "MAX(CASE WHEN field_name='scheduled_burn_original_accel_m_s2' THEN value_real END) "
            "original_requested_accel_m_s2, "
            "MAX(CASE WHEN field_name='scheduled_burn_original_force_n' THEN value_real END) "
            "original_requested_force_n, "
            "MAX(CASE WHEN field_name='scheduled_burn_original_delta_v_m_s' THEN value_real END) "
            "original_requested_delta_v_m_s, "
            "MAX(CASE WHEN field_name='scheduled_burn_duration_s' THEN value_real END) scheduled_burn_duration_s, "
            "MAX(CASE WHEN field_name='requested_force_n' THEN value_real END) controller_requested_force_n, "
            "MAX(CASE WHEN field_name='scheduled_burn_controller_clipped' THEN value_integer END) "
            "controller_clipped FROM fsw_diagnostic_fields GROUP BY object_id), "
            "dv AS (SELECT object_id, value realized_delta_v_m_s FROM metrics WHERE metric_name='total_dv_m_s'), "
            "object_ids AS (SELECT object_id FROM decisions UNION SELECT object_id FROM realized "
            "UNION SELECT object_id FROM diagnostics) "
            "SELECT ids.object_id, d.decision_samples, d.requested_samples, d.applied_decision_samples, "
            "d.saturated_samples, d.max_requested_accel_km_s2, d.max_applied_accel_km_s2, "
            "g.original_requested_accel_m_s2, g.original_requested_force_n, "
            "g.original_requested_delta_v_m_s, g.scheduled_burn_duration_s, "
            "g.controller_requested_force_n, g.controller_clipped, "
            "r.active_thrust_samples, r.max_realized_accel_km_s2, dv.realized_delta_v_m_s "
            "FROM object_ids ids LEFT JOIN decisions d USING(object_id) LEFT JOIN diagnostics g USING(object_id) "
            "LEFT JOIN realized r USING(object_id) LEFT JOIN dv USING(object_id) ORDER BY ids.object_id"
        ),
        allow_empty=True,
        max_vm_steps=500_000,
    ),
    "burn_events": SavedReviewQuery(
        name="burn_events",
        description="Finite-burn start/end and instantaneous maneuver events.",
        sql=(
            "SELECT time_s, object_id, event_type, message FROM events "
            "WHERE event_type IN ('burn_start', 'burn_end', 'impulsive_maneuver') ORDER BY time_s, event_id"
        ),
        allow_empty=True,
    ),
    "controller_decisions": SavedReviewQuery(
        name="controller_decisions",
        description="Compact controller decisions, requested/applied commands, and runtime gate status.",
        sql=(
            "SELECT time_s, object_id, orbit_controller, attitude_controller, mission_strategy, "
            "mission_execution, requested_accel_norm_km_s2, applied_accel_norm_km_s2, "
            "burn_requested, burn_applied, saturated, deadline_missed "
            "FROM controller_decisions ORDER BY time_s, object_id, decision_index"
        ),
    ),
    "mission_mode_timeline": SavedReviewQuery(
        name="mission_mode_timeline",
        description="Mission phase and executive-mode timeline by object.",
        sql=(
            "SELECT time_s, object_id, mission_strategy, mission_execution, mission_phase, executive_mode "
            "FROM mission_modes ORDER BY time_s, object_id, decision_index"
        ),
    ),
    "mission_transitions": SavedReviewQuery(
        name="mission_transitions",
        description="Mission-executive transitions and trigger evidence.",
        sql=(
            "SELECT time_s, object_id, from_mode, to_mode, trigger, reason, detail_json "
            "FROM mission_transitions ORDER BY time_s, object_id, decision_index"
        ),
        allow_empty=True,
    ),
    "command_gate_activity": SavedReviewQuery(
        name="command_gate_activity",
        description="Burn requests suppressed or limited by alignment, fuel, actuator, or deadline gates.",
        sql=(
            "SELECT time_s, object_id, burn_requested, burn_applied, alignment_error_rad, alignment_ok, "
            "fuel_depleted, actuator_limited, deadline_missed, gate_reason FROM command_gates "
            "WHERE gate_reason IS NOT NULL OR burn_requested != burn_applied "
            "ORDER BY time_s, object_id, decision_index"
        ),
        allow_empty=True,
    ),
    "fsw_invocation_summary": SavedReviewQuery(
        name="fsw_invocation_summary",
        description="GNC v2 stack identity, invocation count, inputs, commands, and telemetry by satellite.",
        sql=(
            "SELECT object_id, stack_id, stack_version, COUNT(*) AS invocation_count, "
            "SUM(input_count) AS input_count, SUM(command_count) AS command_count, "
            "SUM(telemetry_count) AS telemetry_count FROM fsw_invocations "
            "GROUP BY object_id, stack_id, stack_version ORDER BY object_id"
        ),
        allow_empty=True,
    ),
    "fsw_diagnostic_field_inventory": SavedReviewQuery(
        name="fsw_diagnostic_field_inventory",
        description=(
            "Bounded diagnostic topic/field inventory with sample counts and time bounds; use the returned "
            "field_name in a selective follow-up query instead of scanning full diagnostic envelopes."
        ),
        sql=(
            "SELECT object_id, topic, field_name, unit, value_kind, COUNT(*) AS samples, "
            "MIN(generated_time_ns) AS first_time_ns, MAX(generated_time_ns) AS last_time_ns "
            "FROM fsw_diagnostic_fields GROUP BY object_id, topic, field_name, unit, value_kind "
            "ORDER BY object_id, topic, field_name"
        ),
        allow_empty=True,
        max_vm_steps=750_000,
    ),
    "fsw_sensor_deliveries": SavedReviewQuery(
        name="fsw_sensor_deliveries",
        description="Delivered GNC v2 measurement packets and their source/delivery times.",
        sql=(
            "SELECT object_id, invocation_id, packet_source_id, packet_boot_id, packet_sequence, "
            "source_time_ns, delivery_time_ns, schema FROM fsw_input_events "
            "WHERE kind = 'measurement' ORDER BY object_id, delivery_time_ns, packet_source_id, packet_sequence"
        ),
        allow_empty=True,
    ),
    "actuator_command_chain": SavedReviewQuery(
        name="actuator_command_chain",
        description="GNC v2 command issue, receipt disposition, and physical realization chain.",
        sql=(
            "SELECT c.object_id, c.invocation_id, c.actuator_id, c.command_source_id, "
            "c.command_boot_id, c.command_sequence, c.issued_time_ns, c.not_before_ns, c.expires_at_ns, "
            "r.received_time_ns, r.disposition, a.interval_start_ns, a.interval_end_ns, "
            "a.demand_mode, a.saturated FROM actuator_commands c "
            "LEFT JOIN actuator_command_receipts r ON r.object_id = c.object_id "
            "AND r.command_source_id = c.command_source_id AND r.command_boot_id = c.command_boot_id "
            "AND r.command_sequence = c.command_sequence LEFT JOIN actuator_realization a "
            "ON a.object_id = c.object_id AND a.command_source_id = c.command_source_id "
            "AND a.command_boot_id = c.command_boot_id AND a.command_sequence = c.command_sequence "
            "ORDER BY c.object_id, c.issued_time_ns, c.command_sequence, a.interval_start_ns"
        ),
        allow_empty=True,
    ),
    "fsw_deadline_misses": SavedReviewQuery(
        name="fsw_deadline_misses",
        description="GNC v2 task releases that exceeded their modeled execution budget.",
        sql=(
            "SELECT object_id, invocation_id, task_id, release_time_ns, modeled_execution_duration_ns, "
            "execution_budget_ns FROM fsw_task_timing WHERE deadline_missed = 1 "
            "ORDER BY object_id, release_time_ns, task_id"
        ),
        allow_empty=True,
    ),
    "safety_requirement_status": SavedReviewQuery(
        name="safety_requirement_status",
        description="Recorded post-run status for every specified flight-software safety requirement.",
        sql=(
            "SELECT object_id, invocation_id, requirement_id, satisfied, source, detail_json "
            "FROM safety_requirement_evidence ORDER BY object_id, requirement_id, invocation_id"
        ),
        allow_empty=True,
    ),
    "fsw_checkpoint_summary": SavedReviewQuery(
        name="fsw_checkpoint_summary",
        description="GNC v2 checkpoint identities and state hashes without exposing opaque state bytes.",
        sql=(
            "SELECT object_id, invocation_id, stack_id, stack_version, state_hash_sha256 "
            "FROM fsw_snapshots ORDER BY object_id, invocation_id"
        ),
        allow_empty=True,
    ),
    "event_log": SavedReviewQuery(
        name="event_log",
        description="Complete ordered termination and runtime event log.",
        sql=(
            "SELECT time_s, object_id, event_type, severity, message FROM events "
            "ORDER BY time_s, event_id"
        ),
        allow_empty=True,
    ),
    "mission_recovery_summary": SavedReviewQuery(
        name="mission_recovery_summary",
        description="Mission-recovery delta-v, time, propellant, and method for configured recovery analysis.",
        sql=(
            "SELECT object_id, goal, method, recovery_available, recovery_delta_v_m_s, "
            "recovery_time_s, recovery_time_basis, propellant_kg, disturbance_delta_v_m_s, "
            "slot_recovery_found, slot_recovery_time_s, best_slot_time_s "
            "FROM mission_recovery_summary"
        ),
    ),
    "mission_recovery_elements": SavedReviewQuery(
        name="mission_recovery_elements",
        description="Initial and final orbital elements used by configured mission-recovery analysis.",
        sql=(
            "SELECT object_id, state_label, a_km, ecc, inc_deg, raan_deg, argp_deg, "
            "true_anomaly_deg FROM mission_recovery_elements ORDER BY object_id, state_label"
        ),
    ),
    "mission_recovery_candidates": SavedReviewQuery(
        name="mission_recovery_candidates",
        description=(
            "Original-orbit analytical baselines and Orbit Transfer Planner candidates ranked by feasibility, "
            "delta-v, and time."
        ),
        sql=(
            "SELECT candidate_id, object_id, goal, source, source_family, target_basis, "
            "transfer_type, planned_delta_v_m_s, planned_time_s, departure_wait_s, "
            "time_of_flight_s, propellant_kg, feasible, verified, recommended_modes_json "
            "FROM mission_recovery_candidates ORDER BY feasible DESC, planned_delta_v_m_s, planned_time_s"
        ),
    ),
    "mission_recovery_burns": SavedReviewQuery(
        name="mission_recovery_burns",
        description="Burn sequence rows for original-orbit baselines and Orbit Transfer Planner candidates.",
        sql=(
            "SELECT candidate_id, burn_index, start_time_s, duration_s, frame, axis, delta_v_m_s "
            "FROM mission_recovery_burns ORDER BY candidate_id, burn_index"
        ),
    ),
    "ground_access_summary": SavedReviewQuery(
        name="ground_access_summary",
        description=(
            "Access sample counts, left-endpoint credited duration, minimum range, and maximum elevation "
            "by station/object."
        ),
        sql=(
            "WITH samples AS (SELECT *, LEAD(time_s) OVER (PARTITION BY station_id, object_id "
            "ORDER BY sample_index) AS next_time_s FROM ground_access) "
            "SELECT station_id, object_id, COUNT(*) AS samples, SUM(access) AS access_samples, "
            "SUM(CASE WHEN access = 1 THEN COALESCE(next_time_s - time_s, 0.0) ELSE 0.0 END) "
            "AS credited_access_duration_s, MIN(range_km) AS min_range_km, "
            "MAX(elevation_deg) AS max_elevation_deg FROM samples "
            "GROUP BY station_id, object_id ORDER BY station_id, object_id"
        ),
    ),
    "ground_access_no_access_reasons": SavedReviewQuery(
        name="ground_access_no_access_reasons",
        description="No-access reason counts by station/object.",
        sql=(
            "SELECT station_id, object_id, reason, COUNT(*) AS samples FROM ground_access "
            "WHERE access = 0 GROUP BY station_id, object_id, reason "
            "ORDER BY station_id, object_id, samples DESC"
        ),
    ),
    "ground_access_windows": SavedReviewQuery(
        name="ground_access_windows",
        description=(
            "Access windows with sampled bounds, explicit first-no-access boundary, credited duration, "
            "extrema, refinement status, and run-boundary censoring."
        ),
        sql=(
            "SELECT station_id, object_id, window_index, start_s, end_s, duration_s, "
            "last_access_sample_s, first_no_access_sample_s, sample_span_s, credited_duration_s, "
            "start_censored, end_censored, minimum_range_km, maximum_elevation_deg, refinement_status, "
            "boundary_semantics "
            "FROM ground_access_windows ORDER BY station_id, object_id, window_index"
        ),
        allow_empty=True,
    ),
    "coverage_summary": SavedReviewQuery(
        name="coverage_summary",
        description=(
            "Whole-Earth coverage sample count, instantaneous extrema, time-weighted mean, "
            "and ever-covered fraction by analysis."
        ),
        sql=(
            "SELECT s.analysis_id, s.source_object_id, "
            "json_extract(s.summary_json, '$.sensor_id') AS sensor_id, "
            "json_extract(s.summary_json, '$.order') AS grid_order, "
            "COUNT(c.sample_index) AS samples, "
            "MIN(c.instantaneous_covered_fraction) AS minimum_fraction, "
            "MAX(c.instantaneous_covered_fraction) AS maximum_fraction, "
            "json_extract(s.summary_json, '$.time_weighted_mean_covered_fraction') "
            "AS time_weighted_mean_covered_fraction, "
            "json_extract(s.summary_json, '$.ever_covered_fraction') AS ever_covered_fraction "
            "FROM coverage_summary s LEFT JOIN coverage_samples c USING (analysis_id) "
            "GROUP BY s.analysis_id, s.source_object_id, s.summary_json "
            "ORDER BY s.analysis_id"
        ),
    ),
    "coverage_transition_summary": SavedReviewQuery(
        name="coverage_transition_summary",
        description="Coverage acquisition/loss counts by refinement disposition.",
        sql=(
            "SELECT analysis_id, transition_kind, disposition, COUNT(*) AS transitions "
            "FROM coverage_transitions GROUP BY analysis_id, transition_kind, disposition "
            "ORDER BY analysis_id, transition_kind, disposition"
        ),
        allow_empty=True,
    ),
    "directed_link_summary": SavedReviewQuery(
        name="directed_link_summary",
        description=(
            "Directed-link sampled availability, range, and margin extrema by analysis."
        ),
        sql=(
            "SELECT s.analysis_id, s.link_id, s.tx_object_id, s.rx_object_id, "
            "s.tx_endpoint_kind, s.rx_endpoint_kind, s.tx_terminal_parent_frame, s.rx_terminal_parent_frame, "
            "COUNT(l.sample_index) AS samples, SUM(l.available) AS available_samples, "
            "MIN(l.range_km) AS minimum_range_km, MAX(l.range_km) AS maximum_range_km, "
            "MIN(l.margin_db) AS minimum_margin_db, MAX(l.margin_db) AS maximum_margin_db, "
            "json_extract(s.summary_json, '$.available_duration_s') AS available_duration_s, "
            "json_extract(s.summary_json, '$.estimated_delivered_data_bits') "
            "AS estimated_delivered_data_bits "
            "FROM link_summary s LEFT JOIN link_samples l USING (analysis_id) "
            "GROUP BY s.analysis_id, s.link_id, s.tx_object_id, s.rx_object_id, "
            "s.tx_endpoint_kind, s.rx_endpoint_kind, s.tx_terminal_parent_frame, s.rx_terminal_parent_frame, "
            "s.summary_json "
            "ORDER BY s.analysis_id"
        ),
    ),
    "directed_link_windows": SavedReviewQuery(
        name="directed_link_windows",
        description="Directed-link availability windows with margin and refinement evidence.",
        sql=(
            "SELECT analysis_id, interval_index, start_s, end_s, duration_s, "
            "minimum_margin_db, mean_margin_db, maximum_margin_db, "
            "estimated_delivered_data_bits, acquisition_disposition, loss_disposition "
            "FROM link_windows ORDER BY analysis_id, interval_index"
        ),
        allow_empty=True,
    ),
    "attitude_rates_first_last": SavedReviewQuery(
        name="attitude_rates_first_last",
        description="First and final angular-rate samples by object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.time_s, s.omega_x_rad_s, s.omega_y_rad_s, "
            "s.omega_z_rad_s FROM object_state s JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "attitude_error_first_last": SavedReviewQuery(
        name="attitude_error_first_last",
        description="First/final attitude error for each object, from retained desired quaternions or controller telemetry.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(rowid) AS first_rowid, "
            "MAX(rowid) AS last_rowid FROM attitude_error GROUP BY object_id) "
            "SELECT a.object_id, a.sample_index, a.time_s, a.pointing_error_deg, "
            "a.quat_error_angle_deg FROM attitude_error a JOIN bounds b "
            "ON a.object_id = b.object_id AND a.rowid IN (b.first_rowid, b.last_rowid) "
            "ORDER BY a.object_id, a.time_s, a.rowid"
        ),
        allow_empty=True,
    ),
    "attitude_state_first_last": SavedReviewQuery(
        name="attitude_state_first_last",
        description="First and final quaternion and angular-rate samples by object.",
        sql=(
            "WITH bounds AS (SELECT object_id, MIN(sample_index) AS first_i, "
            "MAX(sample_index) AS last_i FROM object_state GROUP BY object_id) "
            "SELECT s.object_id, s.time_s, s.quat_w, s.quat_x, s.quat_y, s.quat_z, "
            "s.omega_x_rad_s, s.omega_y_rad_s, s.omega_z_rad_s FROM object_state s "
            "JOIN bounds b ON s.object_id = b.object_id "
            "AND s.sample_index IN (b.first_i, b.last_i) ORDER BY s.object_id, s.sample_index"
        ),
    ),
    "workflow_metadata": SavedReviewQuery(
        name="workflow_metadata",
        description="Workflow review type, scenario, status, generation time, schema version, and source config.",
        sql=(
            "SELECT workflow_type, scenario_name, title, status, generated_utc, "
            "review_schema_version, source_config FROM workflow_metadata"
        ),
    ),
    "workflow_artifacts": SavedReviewQuery(
        name="workflow_artifacts",
        description="Workflow review artifact inventory.",
        sql="SELECT artifact_key, artifact_type, path FROM workflow_artifacts ORDER BY artifact_type, artifact_key",
    ),
    "controller_bench_runs": SavedReviewQuery(
        name="controller_bench_runs",
        description="Controller-bench run pass/fail rows and output directories.",
        sql=(
            "SELECT variant_name, case_name, passed, failure_count, output_dir "
            "FROM bench_runs ORDER BY variant_name, case_name"
        ),
    ),
    "controller_bench_leaderboard": SavedReviewQuery(
        name="controller_bench_leaderboard",
        description="Controller-bench leaderboard rows by objective and metric.",
        sql=(
            "SELECT kind, objective, metric, rank, variant_name, value, samples "
            "FROM bench_leaderboard ORDER BY kind, objective, metric, rank"
        ),
    ),
    "controller_bench_failures": SavedReviewQuery(
        name="controller_bench_failures",
        description="Controller-bench objective failures and suggestions.",
        sql=(
            "SELECT variant_name, case_name, objective, metric, reason, failure_mode, suggestion "
            "FROM bench_failures ORDER BY variant_name, case_name, objective"
        ),
    ),
    "sensitivity_rankings": SavedReviewQuery(
        name="sensitivity_rankings",
        description="Sensitivity parameters ranked by effect size.",
        sql=(
            "SELECT parameter_path, metric_path, method, effect_size, rank "
            "FROM sensitivity_rankings ORDER BY rank, parameter_path, metric_path"
        ),
    ),
    "sensitivity_runs": SavedReviewQuery(
        name="sensitivity_runs",
        description="Sensitivity generated-run status and changed parameter rows.",
        sql=(
            "SELECT run_id, status, parameter_path, parameter_value, output_dir "
            "FROM sensitivity_runs ORDER BY run_id"
        ),
    ),
    "campaign_runs": SavedReviewQuery(
        name="campaign_runs",
        description="Monte Carlo campaign runs with pass/fail and key metrics.",
        sql=(
            "SELECT iteration, passed, closest_approach_km, duration_s, total_dv_m_s, output_dir "
            "FROM campaign_runs ORDER BY iteration"
        ),
    ),
    "campaign_metrics": SavedReviewQuery(
        name="campaign_metrics",
        description="Flattened Monte Carlo campaign metrics by iteration.",
        sql="SELECT iteration, metric_name, metric_value FROM campaign_metrics ORDER BY iteration, metric_name",
    ),
    "validation_benchmarks": SavedReviewQuery(
        name="validation_benchmarks",
        description="Validation benchmark pass/fail rows.",
        sql=(
            "SELECT benchmark_name, kind, passed, duration_s, output_dir "
            "FROM validation_benchmarks ORDER BY benchmark_name"
        ),
    ),
    "od_phase9_pilots": SavedReviewQuery(
        name="od_phase9_pilots",
        description="Productized OD pilot capability, claim level, readiness, and evidence packet paths.",
        sql=(
            "SELECT case_id, capability_id, claim_level, status, passed, evidence_packet_path "
            "FROM od_phase9_pilots ORDER BY case_id"
        ),
        maturity="supported",
    ),
    "sequential_od_summary": SavedReviewQuery(
        name="sequential_od_summary",
        description="Sequential OD estimator, observation decisions, and evidence state.",
        sql=(
            "SELECT object_id, estimator, observation_count, accepted_count, rejected_count, evidence_status "
            "FROM sequential_od_run"
        ),
        maturity="supported",
    ),
    "integrated_relative_od_models": SavedReviewQuery(
        name="integrated_relative_od_models",
        description="Integrated relative OD model fit, holdout, and sequential comparison rows.",
        sql=(
            "SELECT model, fit_position_rms_m, holdout_position_rms_m, batch_solver_success, "
            "sequential_position_rms_m FROM relative_od_models ORDER BY model"
        ),
        maturity="supported",
    ),
}


def get_saved_review_query(name: str) -> SavedReviewQuery | None:
    return SAVED_REVIEW_QUERIES.get(str(name or "").strip())


def list_saved_review_queries() -> list[SavedReviewQuery]:
    return [SAVED_REVIEW_QUERIES[name] for name in sorted(SAVED_REVIEW_QUERIES)]
