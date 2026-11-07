# Payload And Artifact Contract

This document defines the 0.5 contract for simulation payloads, summaries, and
output artifacts. It focuses on single-run payloads and the public review
surface. Batch/campaign payloads are included only at a high level because their
full schemas remain Pro/private maturity work.


## Stability Level

Stable enough to rely on:

- single-run `summary` presence,
- core summary fields listed below,
- time/truth/belief/thrust/torque history keys,
- plot and animation artifact maps in summary,
- public API convenience properties that wrap these fields.

Still maturing:

- full-log debug structure,
- controller debug record contents,
- campaign and sensitivity report schemas,
- AI report metadata schemas,
- exact artifact filenames beyond published manifest mappings.


## Single-Run Payload Shape

A single-run payload should include these top-level keys when applicable:

- `summary`
- `time_s`
- `truth_by_object`
- `target_reference_orbit_truth`
- `belief_by_object`
- `applied_thrust_by_object`
- `applied_torque_by_object`
- `desired_attitude_by_object`
- `knowledge_by_observer`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`
- `bridge_events_by_object`
- `controller_debug_by_object`
- `command_decisions_by_object`
- `rocket_throttle_cmd`
- `rocket_metrics`
- `reentry_metrics_by_object`
- `orbital_analysis` when coverage or directed-link postprocessing is enabled

Consumers should treat `summary` as the most stable review surface. Detailed
histories and debug maps are available for analysis, but fields not documented
here should be considered lower stability until tests and docs promote them.

`command_decisions_by_object` is the compact standard GNC record and is present
even when detailed controller debug is disabled. Rows include controller and
mission identities, requested/applied command vectors and norms, mission mode,
intent-field provenance, saturation, deadline, fuel/actuator limits, alignment,
and gate reason. `controller_debug_by_object` remains the larger opt-in
engineering record with belief and raw component details.


## Summary Contract

The single-run `summary` should include:

- `scenario_name`
- `scenario_description`
- `objects`
- `samples`
- `dt_s`
- `duration_s`
- `terminated_early`
- `termination_reason`
- `termination_time_s`
- `termination_object_id`
- `rocket_insertion_achieved`
- `rocket_insertion_time_s`
- `target_reference_orbit_enabled`
- `reference_object_id`
- `primary_object_pair`
- `thrust_stats`
- `attitude_guardrail_stats`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`
- `ground_station_access_summary`
- `reentry_summary_by_object`
- `plot_outputs`
- `animation_outputs`
- `review_outputs`
- `orbital_analysis` when configured, with coverage/directed-link summaries,
  semantic/input evidence digests, schema/product identities, and artifact
  descriptors
- `mission_recovery` when configured, with assessment/planner status,
  selected recommendation, candidate/burn counts, limits, and warnings
- `output_index_md`

Compatibility expectations:

- `scenario_name` is a string.
- `objects` is a sorted list of participating object IDs.
- `samples` is the number of retained time samples.
- `duration_s` is the final retained simulation time, not necessarily the
  requested duration when a run terminates early.
- `terminated_early` is the authoritative early-termination flag.
- Termination detail fields may be `null` for nominal completion.
- `thrust_stats` is keyed by object ID.
- `reference_object_id` is the default object used for relative frames and
  plots when one is available.
- `primary_object_pair` is the default pair for relative metrics and plots; it
  should be treated as object IDs, not fixed roles.
- `plot_outputs` and `animation_outputs` are maps from artifact/figure IDs to
  paths or artifact descriptors.


## History Arrays

The following histories are keyed by object ID unless otherwise noted:

- `truth_by_object`
- `belief_by_object`
- `applied_thrust_by_object`
- `applied_torque_by_object`
- `desired_attitude_by_object`

Expected conventions:

- `time_s` is a one-dimensional list of sample times in seconds.
- Object histories should have the same first dimension as `time_s`.
- Truth state arrays use the project state convention documented by code and
  tests: position, velocity, attitude quaternion, angular rate, and mass where
  available.
- Applied thrust is in `km/s^2`.
- Applied torque is in `N*m`.
- Missing or inactive values may be represented by empty arrays, omitted maps,
  or `NaN` samples depending on the object lifecycle.

Consumers should not infer object participation solely from a history key; use
`summary.objects` for the stable list of participating objects.


## Knowledge And Estimation Artifacts

Knowledge fields:

- `knowledge_by_observer`
- `knowledge_detection_by_observer`
- `knowledge_consistency_by_observer`

Conventions:

- Knowledge is observer-owned and target-indexed.
- Detection summaries describe access/update behavior.
- Consistency summaries may include metrics such as update rate, NIS/NEES-style
  values, and position-error statistics when configured.

These fields are important validation surfaces, but exact nested metric sets may
grow. Validation harness checks should use explicit metric paths so schema
expectations are testable.


## Rocket Metrics

Rocket-specific fields may include:

- `rocket_throttle_cmd`
- `rocket_metrics.stage_index`
- `rocket_metrics.q_dyn_pa`
- `rocket_metrics.mach`
- `rocket_metrics.throttle_cmd`
- `rocket_metrics.altitude_km`
- `rocket_metrics.speed_km_s`
- `rocket_metrics.vertical_speed_km_s`
- `rocket_metrics.horizontal_speed_km_s`
- `rocket_metrics.flight_path_angle_deg`
- `rocket_metrics.apoapsis_alt_km` (apogee altitude)
- `rocket_metrics.periapsis_alt_km` (perigee altitude)
- `rocket_metrics.eccentricity`
- `rocket_metrics.alpha_deg`
- `rocket_metrics.beta_deg`
- `rocket_metrics.tvc_gimbal_deg`
- `rocket_metrics.aero_force_n`
- `rocket_metrics.aero_moment_nm`
- `rocket_metrics.thrust_to_weight`
- `rocket_metrics.propellant_remaining_kg`
- `rocket_metrics.propellant_remaining_fraction`
- `rocket_metrics.guidance_phase_code`
- `summary.rocket_metrics_summary`

Rocket metrics are present only when a rocket object participates. Consumers
should handle absent or empty rocket metrics for satellite-only scenarios.
The exact rocket metric set may grow as the rocket GNC stack matures; consumers
should prefer explicit key checks over assuming a fixed list.


## Re-Entry Metrics

Re-entry fields may include:

- `reentry_metrics_by_object.<object_id>.active`
- `reentry_metrics_by_object.<object_id>.altitude_km`
- `reentry_metrics_by_object.<object_id>.density_kg_m3`
- `reentry_metrics_by_object.<object_id>.relative_speed_m_s`
- `reentry_metrics_by_object.<object_id>.dynamic_pressure_pa`
- `reentry_metrics_by_object.<object_id>.drag_decel_m_s2`
- `reentry_metrics_by_object.<object_id>.lift_accel_m_s2`
- `reentry_metrics_by_object.<object_id>.lift_to_drag`
- `reentry_metrics_by_object.<object_id>.g_load`
- `reentry_metrics_by_object.<object_id>.heat_rate_w_m2`
- `reentry_metrics_by_object.<object_id>.heat_load_j_m2`
- `summary.reentry_summary_by_object`

Re-entry metrics are present only when `simulator.dynamics.reentry.enabled` is
true. The metrics are sampled after each propagation step from the current
state. `active` means the object is currently below the configured re-entry
altitude threshold, while `summary.reentry_summary_by_object` preserves
ever-entered state, episode count, latest exit time, peak lift acceleration,
lift-to-drag ratio, and cumulative heat load.
During satellite burns, the burn influences the sampled state, but
`drag_decel_m_s2` and `g_load` remain aerodynamic drag-load estimates rather
than total acceleration.


## Ground Station Access

Ground-station fields:

- `ground_station_access`
- `ground_station_access_summary`
- `ground_station_measurements`
- `ground_station_measurement_summary`
- `summary.ground_station_access_summary`
- `summary.ground_station_measurement_summary`
- `summary.ground_station_access_report_outputs`

Conventions:

- Ground stations are passive observers configured through scenario YAML.
- Access histories are station-indexed, then target-indexed.
- Per-sample histories include `access`, `line_of_sight`, `range_km`,
  `elevation_deg`, and a diagnostic `reason`.
- Summaries include access sample count, access fraction, interval-based access
  duration, first/last access time, minimum range, and maximum elevation.
- Human-readable access reports are written in both satellite-oriented and
  ground-station-oriented Markdown forms when access data exists.
- Access report AOS/LOS rows use UTC. The report epoch is
  `simulator.initial_jd_utc` when supplied, otherwise
  `2026-01-01T00:00:00Z`.
- When enabled, measurement histories are station-indexed, then target-indexed.
  V0 measurement rows include `time_s`, optional `jd_utc`, `azimuth_deg`,
  `elevation_deg`, `range_km`, optional `range_rate_km_s`, component names,
  configured sigmas, matching truth geometry fields for validation, and
  `valid`/`validity_reason` disposition fields. Seeded Gaussian draws remain
  raw evidence, including physically impossible range or angular values; rows
  with `valid: false` are retained for audit but are rejected before
  observation-packet conversion or receiver delivery. `measurement_count`
  counts raw rows and `usable_measurement_count` counts rows eligible for
  downstream conversion.
- Ground-station access does not modify truth, belief, knowledge, control, or
  termination behavior.
- Ground-station measurements are synthetic observation evidence. They do not
  model RF link budgets, weather, scheduling, calibration, association, or real
  sensor processing.


## Artifacts On Disk

Stats artifacts:

- `index.md` is the human-readable start-here file for an output directory.
  It summarizes the workflow, scenario, key results, suggested review order,
  and artifact inventory.
- `master_run_summary.json` is written when `outputs.stats.save_json` is true.
- `master_run_log.json` is written when `outputs.stats.save_full_log` is true.
- `master_run_history.npz` is written when `outputs.stats.save_history_npz` is
  true. It contains NumPy history arrays plus a `manifest_json` entry that maps
  archive keys back to payload paths.
- `review/run.sqlite` and `review/schema.json` are written when
  `outputs.review.enabled` is true.
- A capsule-only store uses logical `review/run.sqlite` evidence represented by
  `review/run.sqlite.gz` plus `review/evidence_capsule.json`.

Plot and animation artifacts:

- Plot artifacts are represented in `summary.plot_outputs`.
- Animation artifacts are represented in `summary.animation_outputs`.
- Consumers should read artifact maps instead of constructing filenames from
  assumptions.
- Orbital-analysis coverage and directed-link descriptors are exposed through
  `SimulationResult.artifacts["orbital_analysis"]` and retain their own
  versioned schema/product metadata.
- Mission recovery may add `plot_outputs["mission_recovery_trade_space"]`;
  use the dedicated recovery review tables for candidate and burn evidence.

Review artifacts:

- Review-store artifact paths are represented in `summary.review_outputs` when
  enabled and successfully written.
- The first single-run review store is a SQLite database intended for output
  review and SQL-like inspection, not a replacement for the authoritative
  scenario config or deterministic simulator.

Output directories:

- `outputs.output_dir` controls where artifacts are written.
- Tests and validation workflows should prefer temporary directories unless
  intentionally checking repo-level example outputs.


## Public API Wrappers

`SimulationResult` provides convenience accessors over payload fields:

- `summary`
- `time_s`
- `object_ids`
- `reference_object_id`
- `primary_pair`
- `truth`
- `belief`
- `applied_thrust`
- `applied_torque`
- `knowledge`
- `ground_station_access`
- `artifacts`
- `output_dir`
- `metrics`
- `review()`
- `evidence_manifest()`
- `snapshot(step_index)`
- `relative_state(...)`
- `range_between(...)`
- `min_range(...)`
- `time_of_min_range(...)`

`evidence_manifest()["review"]` distinguishes a physical plain database
(`db_exists`) from logical evidence (`evidence_exists`). It reports storage,
logical SHA-256, compressed/capsule paths, and their existence so a capsule-only
store is not misreported as absent.
- `collision_event(...)`
- `keepout_violations(...)`
- `to_records(...)`
- `to_dataframe(...)`
- `SimulationWorkspace.validate(...)`
- `SimulationWorkspace.validate_report(...)`
- `SimulationWorkspace.run(...)`
- `SimulationWorkspace.save_config(...)`

`ScenarioArtifact` is the public API wrapper for YAML-as-artifact workflows. It
supports `from_dict(...)`, `from_yaml(...)`, `to_yaml_text()`, `write(...)`,
`to_artifact_dict()`, `validate(...)`, `validate_report(...)`, and `run(...)`.
`to_dict()` returns the normalized engine config; `to_artifact_dict()` and
`write(...)` render a cleaner review artifact that omits compatibility/default
noise while preserving reload/run behavior. The artifact path is intended for
agent, notebook, and app authoring flows that still need a durable YAML scenario
record.

`ScenarioBuilder` is the small public authoring helper for common
single-satellite, basic target/chaser relative-state, and ground-station
scenarios. Builder methods should remain domain-facing and should emit
`ScenarioArtifact` rather than bypassing scenario YAML normalization. The
authoring helpers live in `sim.scenarios` and are re-exported through `sim` and
`sim.api`.

These wrappers are part of the intended public API. If payload keys change,
wrappers should either preserve compatibility or receive release-note coverage
and focused tests.

Private Pro workflow helpers such as controller-bench execution, Monte Carlo
custom-metric runners, parameter sweeps, and AI report/config helpers are
documented separately in the private Pro Python API guide.


## Batch And Campaign Payloads

Batch payloads may include:

- `monte_carlo`
- `analysis`
- `runs`
- `aggregate_stats`
- `commander_brief`
- `artifacts`
- baseline comparisons,
- sensitivity rankings,
- AI report artifacts.

These are Pro/private workflow surfaces in 0.4. They are real and tested, but
their complete schema is not yet a public-core contract. Campaign and benchmark
schema stabilization should happen in a dedicated benchmark/campaign contract.


## Compatibility Rules

Changes should update this contract when they alter:

- stable summary field names or meanings,
- top-level single-run payload keys,
- history units,
- artifact-map semantics,
- public API wrapper behavior,
- termination metadata semantics.

For new analysis tools, prefer adding fields over changing existing meanings.
If an existing field must change, add migration notes and tests.


## Known Gaps

- No generated JSON schema exists for payloads.
- Campaign, sensitivity, benchmark, and AI report payloads need dedicated
  contracts.
- Debug records are intentionally lower stability than summaries.
- Some artifact maps still need stronger descriptor normalization.
