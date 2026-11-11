# Scenario YAML Contract

This document defines the OEL 1.x compatibility contract for scenario YAML files.
It complements the user-facing guide in `docs/scenario-yaml.md` by stating what
scenario authors, examples, tests, agents, and migration tools may rely on.

The contract applies to public local single-run and bounded analysis/campaign
scenarios. Hosted orchestration, provider-driven AI, restricted validation
inputs, and other private-service fields are called out explicitly.


## Stability Level

The [OEL 1.x compatibility policy](../compatibility-policy.md) governs this
contract. Minor releases may add compatible optional fields. Breaking changes
require a versioned successor while preserving the supported 1.x path.
Contract changes and documented correctness fixes require:

- a release-note entry,
- an update to this contract,
- a focused regression test,
- migration guidance and a compatibility shim when introducing a successor.

Stable enough to rely on:

- top-level section names,
- strict boolean parsing,
- object preset resolution order,
- plugin pointer shape,
- timing-grid validation,
- output section shape,
- acceptance of documented local analysis/campaign workflows without license
  activation, and rejection of unavailable private-service fields.

Still maturing:

- migration tooling for all historical config shapes,
- exhaustive invalid-combination diagnostics.


## Top-Level Sections

Recognized top-level sections:

- `schema_version` (`oel.scenario.v1` for new configs; unversioned historical
  files are admitted as `oel.scenario.legacy.v0` during the support window)
- `scenario_name`
- `scenario_description`
- `metadata`
- `objects`
- `ground_stations`
- `simulator`
- `outputs`
- `analysis`

Object sections:

- `objects` is the canonical object map for new configs.
- Object IDs are user-facing names. Conventional IDs such as `rocket`,
  `chaser`, and `target` are supported but are not the only valid IDs.
- Top-level `rocket`, `chaser`, and `target` sections are no longer accepted in
  scenario YAML. Use `objects.<id>` for every participant.
- Disabled objects may be omitted or set with `enabled: false`.
- Enabled object entries participate in runtime creation, validation, and
  output histories.

Ground station sections:

- `ground_stations` may be omitted, a list of station mappings, or a mapping
  from station ID to station mapping.
- Ground stations are passive scene observers. They do not participate in
  dynamics, control, estimation, or knowledge ownership.
- Each enabled station requires `lat_deg` and `lon_deg`, and may provide
  `alt_km`, `min_elevation_deg`, and `max_range_km`.
- Access is defined as geometric line of sight, elevation at least
  `min_elevation_deg`, and range no greater than `max_range_km` when configured.
- Ground stations may opt into synthetic measurement rows with a
  `measurements` mapping. Supported v0 measurement types are `az_el_range` and
  `az_el_range_rate`, with optional `update_cadence_s`, deterministic `seed`,
  and scalar range/range-rate/angle noise sigmas.
- Ground-station measurements are generated evidence and do not modify
  dynamics, control, estimation, knowledge, termination, or truth histories.

Analysis sections:

- Public local OEL accepts deterministic single-run scenarios and the bounded
  campaign/analysis entry points documented in `docs/v1-local-workflows.md`.
- Private development and Hosted workspaces may add orchestration, provider,
  or governance layers around those local workflows.
- Generated public entrypoints should reject unsupported private orchestration,
  provider-driven AI, or operational-authority requests with a clear boundary
  message; they should not reject the documented local campaign/analysis
  routes merely because their legacy module names contain `pro`.


## Strict Types

Boolean fields must be YAML booleans:

```yaml
enabled: true
drag: false
```

Quoted boolean-like strings are invalid:

```yaml
enabled: "true"   # invalid
drag: "false"    # invalid
```

Strict boolean parsing applies broadly to fields named or shaped like:

- `enabled`
- `strict`
- `j2`, `j3`, `j4`
- `drag`
- `srp`
- `third_body_moon`
- `third_body_sun`
- `parallel_enabled`
- keys beginning with `use_`, `save_`, `display_`, `print_`, or `require_`

Scenario authors should prefer explicit numeric values for times, distances,
masses, and tolerances. Loader behavior may coerce some values, but config files
should not rely on stringly typed numerics.


## Timing Grid

Required timing rules:

- `simulator.duration_s` must be positive.
- `simulator.dt_s` must be positive.
- `duration_s` must be an integer multiple of `dt_s`.
- `orbit_substep_s`, when present, must be positive, no larger than `dt_s`, and
  divide `dt_s` cleanly.
- `attitude_substep_s`, when present, must be positive, no larger than `dt_s`,
  and divide `dt_s` cleanly.
- When attitude dynamics are enabled, `attitude_substep_s` must be no larger
  than `orbit_substep_s`.

Analysis parameter sweeps should not vary timing fields unless every generated
sample still preserves these grid rules.


## Object Sections

Common object fields:

- `enabled`
- `object_id`
- `kind`
- `role`
- `preset`, `preset_yaml`, or `preset_path`
- `specs`
- `initial_state`
- `reference_orbit`
- `knowledge`
- `flight_software` (satellites)
- rocket guidance/vehicle fields documented by the rocket contract (rockets)

Satellite mission loads use `flight_software.mission_load`. There is no
top-level satellite `mission` field, and bridge adapters use the normalized
`flight_software.module`, `flight_software.class_name`, and
`flight_software.params` fields.

The retired satellite v1 fields `guidance`, `base_guidance`,
`guidance_modifiers`, `orbit_control`, `attitude_control`, `mission_strategy`,
`mission_execution`, `mission_objectives`, and top-level `bridge` are rejected.
Use the canonical complete `flight_software` boundary instead; bridge settings,
when entitled, are nested under that boundary.

`initial_state` must select exactly one orbital initialization form. Current
forms include:

- `position_eci_km` with required `velocity_eci_km_s`,
- `coes`,
- `tle`,
- `ogp_mean_elements`,
- `relative_to_target_ric`, `relative_ric_rect`, or `relative_ric_curv`,
- `relative_to_target_cislunar` or `relative_cislunar`,
- `cr3bp_rotating` or `cr3bp_halo`,
- `launch_lat_deg` launch initialization,
- `source: rocket_deployment` with `deploy_time_s`,
- `source: rocket_insertion`,
- `source: default_circular_earth` for the explicit default/trajectory-only path.

Specialized relative, cislunar/CR3BP, launch, OGP-mean-element, and deployment
forms retain their documented object-kind and public/local/private-service
boundaries. Cartesian position requires Cartesian velocity; auxiliary epoch,
attitude, and deployment metadata do not create a second orbital form.

`coes` must contain at least one finite element. Canonical keys are `a_km`,
`ecc`, `inc_deg`, `raan_deg`, `argp_deg`, and `true_anomaly_deg`. The established
compatibility aliases `semi_major_axis_km`, `e`, `inclination_deg`,
`arg_periapsis_deg`, and `ta_deg` remain accepted. Omitted elements retain the
existing converter defaults: `a_km=7000` and zero for eccentricity and all
angles.

Rocket-deployed satellites may also set `deploy_dv_body_m_s`, a body-frame
deployment delta-v in meters per second. `initialization_delay_s` is optional
and defaults to `0.0`; while the delay is active, the satellite coasts and is
visible to truth/knowledge/history, but its mission and controllers cannot
command thrust or torque.

The `tle` form accepts either `line1`/`line2` or `lines: [line1, line2]`.
TLE checksum validation is enabled by default; set `tle.require_checksum: false`
only for explicitly identified synthetic fixtures.
Optional `tle.propagate_to_initial_epoch: false` keeps the state at the TLE
epoch instead of advancing mean anomaly to `simulator.initial_jd_utc`.
The default TLE initializer samples OGP-SGP4/SDP4 and applies the configured
TEME-to-ECI initialization transform. The legacy two-body mean-element
initializer requires an explicit opt-in. Subsequent propagation uses the configured OEL
numerical special-perturbations force model; continuous OGP general-perturbations
propagation requires object-level `propagation_method: general` with
`general.model: sgp4`.

Satellite `runtime_profile` accepts `flight_software` (the default) or
`trajectory_only`. The default preserves the complete-stack boundary and
normalizes an omitted `flight_software` section to `fsw.passive`.
`trajectory_only` is an explicit dynamics-only posture: the configured ONP/OGP
propagation and force models remain active, but no onboard sensors, knowledge,
flight-software runtime, devices, commands, or FSW evidence are created. A
trajectory-only object must not declare `flight_software` or object-owned
`knowledge`; other objects may still observe its truth state.

Preset merge contract:

- Presets load before local object overrides.
- Local values override preset values.
- Nested dictionaries merge recursively.
- If a local override provides `specs.mass_kg` without `dry_mass_kg` or
  `fuel_mass_kg`, preset dry/fuel masses are ignored for that object so the
  explicit total mass is honored.
- Objects may define `specs.mass_properties.inertia_kg_m2` as a finite 3x3
  body-frame inertia matrix in kg m^2. Strict validation rejects explicitly
  supplied inertia matrices that are non-symmetric, not positive definite, or
  fail principal-moment triangle inequalities. Optional audit metadata includes
  `center_of_mass_body_m`, `inertia_reference_point`, `frame`, `source`, and
  `confidence`.

Preset resolution order:

1. absolute path, when provided,
2. path relative to the scenario YAML file,
3. path relative to the current working directory,
4. path relative to the repository root,
5. name inside `sim/presets/objects`,
6. same name with `.yaml` inside `sim/presets/objects`.


## Algorithm Pointers

Python plugin pointer shape:

```yaml
flight_software:
  module: "my_package.flight_software"
  class_name: "MySatelliteFlightSoftware"
  params: {}
```

Pointer contract:

- `module` must be an importable Python module.
- A class pointer uses `class_name`.
- A function pointer uses `function`, but only pointer roles that explicitly
  allow functions may use it. The complete flight-software boundary requires a
  class implementing `SatelliteFlightSoftware`.
- `params` must be a mapping.
- File-path plugin loading is not part of the scenario YAML contract.

Plugin validation checks importability, symbols, and required callable methods
for supported plugin roles without constructing user plugin instances. Validation
is intentionally structural; it does not prove physical correctness. Plugins
that need richer checks should expose explicit validation hooks rather than
depending on constructor side effects.


## Simulator Section

Common fields:

- `duration_s`
- `dt_s`
- `initial_jd_utc`
- `dynamics`
- `environment`
- `plugin_validation`
- `termination`

Scenario behavior is expressed through object roles, object kinds, dynamics,
controllers, mission modules, bridges, and analysis sections. `scenario_type`
is no longer accepted in scenario YAML.

Dynamics contract:

- Orbit dynamics live under `simulator.dynamics.orbit`.
- Attitude dynamics live under `simulator.dynamics.attitude`.
- Rocket dynamics configuration lives under `simulator.dynamics.rocket`.
- Atmospheric re-entry diagnostics live under `simulator.dynamics.reentry`.
  `enabled` must be a boolean, `begin_altitude_km` must be non-negative, and
  `object_ids` must be a string or list of strings. Vehicle nose geometry belongs
  on object specs under `aero.nose_radius_m`, with `nose_radius_m` and
  `reentry_nose_radius_m` accepted as legacy aliases. Re-entry tracking
  is current-state based: objects are active while below `begin_altitude_km`,
  while summaries preserve whether they ever entered and how many entry episodes
  occurred. Re-entry tracking does not override `orbit.drag`; drag remains the
  force-model toggle. Optional atmospheric steering also depends on `orbit.drag`
  and is configured on satellite object specs under `aero` with `cl`, optional
  `lift_area_m2`, and `lift_axis_body` or `lift_vector_body`. The flat forms
  remain accepted as aliases.
- Shared vehicle aero properties should live under `objects.<id>.specs.aero`.
  Common keys are `reference_area_m2`, `drag_area_m2`, `lift_area_m2`, `cd`,
  `cl`, `nose_radius_m`, `reference_length_m`,
  `lift_axis_body`/`lift_vector_body`, and `cp_offset_body_m`.
  `objects.<id>.specs.aero` owns vehicle physical/aero properties;
  `simulator.dynamics.orbit.drag` owns force-model enablement;
  `simulator.dynamics.reentry` owns diagnostics, limits, and termination; and
  `simulator.dynamics.rocket.aero` owns detailed rocket coefficient refinements.
  Flat compatibility aliases override same-named nested `specs.aero` values
  when both are present. Rocket-specific coefficient refinements under
  `simulator.dynamics.rocket.aero` are still supported and override the shared
  object-level defaults for rocket ascent.
  Aero numeric values must be finite. Areas and `cd` must be non-negative,
  `cl` may be signed, and `nose_radius_m` and `reference_length_m` must be
  positive.
- Re-entry early termination lives under
  `simulator.dynamics.reentry.termination`. `enabled` and
  `terminate_on_entry` must be booleans. Numeric limits must be non-negative:
  `min_altitude_km`, `max_dynamic_pressure_pa`, `max_drag_decel_m_s2`,
  `max_g_load`, `max_heat_rate_w_m2`, and `max_heat_load_j_m2`.
  `termination.by_object` may provide per-object overrides using the same keys;
  unspecified object values inherit the scenario-level termination defaults.
- Time-dependent environment behavior may use `initial_jd_utc` and environment
  ephemeris settings.

Termination contract:

- `termination.earth_impact_enabled` controls Earth-impact termination.
- `termination.earth_radius_km` defines the impact radius when enabled.
- `termination.by_object` may override `earth_impact_enabled` and
  `earth_radius_km` per object. Object overrides inherit the scenario-level
  Earth-impact defaults unless a field is supplied.


## Outputs Section

Common fields:

- `output_dir`
- `mode`
- `stats`
- `plots`
- `animations`
- `monte_carlo`
- `ai_report`
- `review`
- `orbital_analysis`
- `resource_limits`

Output modes:

- `save` is the preferred headless/CI mode.
- `interactive` may open plot windows or require display-capable environments.
- In automation contexts, interactive mode may be coerced to save mode.

Stats contract:

- `stats.save_json` controls `master_run_summary.json`.
- `stats.save_full_log` controls `master_run_log.json`.
- `stats.print_summary` controls console summary printing.

Plot and animation contract:

- Plots and animations are optional artifacts.
- Disabled plots/animations should not be required for simulation correctness.
- Artifact filenames and figure IDs should be consumed through summary artifact
  mappings rather than hard-coded when possible.

Review store contract:

- `review.enabled` controls whether a single-run SQLite review store is written
  under `outputs.output_dir/review/`.
- `review.detail` must be one of `compact`, `standard`, or `full`.
- `review.strict` controls whether review-store writer failures raise or are
  recorded as non-fatal review status while preserving normal artifacts.


## Public Local And Private-Service Fields

Public-core scenarios should use:

- deterministic single-run object sections,
- public complete flight-software stacks or an importable custom complete stack,
- public dynamics settings,
- public output settings,
- curated examples under `examples/configs`.

Public local analysis scenarios may additionally use:

- `analysis.enabled: true`,
- `analysis.study_type: monte_carlo`,
- `analysis.study_type: covariance`,
- controller-benchmark configs,
- bounded local campaign and Controller Bench settings,
- local validation recipes and synthetic fixtures,
- restricted HPOP/MATLAB validation inputs only in authorized development
  workflows,
- integration/SIL bridge workflows.

The generated public export may contain stub modules or boundary errors for
private-service features. Public configs should not require those features to
run.


## Migration Rules

When the scenario contract changes:

- update `docs/scenario-yaml.md` when user-facing authoring guidance changes,
- update this contract when compatibility expectations change,
- add regression tests for old and new behavior where practical,
- update curated configs before release,
- note migration-sensitive changes in `CHANGELOG.md`.

Do not silently reinterpret old fields in a way that changes physics,
controller behavior, or output semantics without tests and release notes.


## Known Gaps

- `schema_version` accepts the current `oel.scenario.v1` and the temporary
  legacy identifier; omission is classified as legacy and unknown versions
  fail closed. A generated full machine-readable schema remains incomplete.
- Not every supported field has a generated machine-readable schema.
- Public/private-service field boundaries are enforced mostly through export
  stubs, docs, config curation, and runtime checks rather than a standalone
  schema compiler.
