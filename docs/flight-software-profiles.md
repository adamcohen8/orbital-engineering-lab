# Flight-Software Use-Case Profiles

OEL's use-case profiles are versioned, prebuilt selections layered over the
complete GNC v2 stack boundary. A profile selects one existing complete stack,
a physical hardware family, an onboard task cadence, and a coherent minimum
set of defaults for a recognizable spacecraft application. Mission-specific
targets remain explicit inputs.

Profiles began as productization groundwork rather than qualification claims.
Maturity attaches to the exact profile version and its declared envelope, not
to every profile that uses the same stack. All 18 exact catalog profiles have a **declared Supported** maturity inside
bounded simulation qualification envelopes. Effective maturity depends on
current qualification evidence for the exact source and profile version.
Missing or stale evidence downgrades the effective status to Experimental.

Use `python -m sim.flight_software list --kind profile --json` or `show` for
the effective status and its separate `declared_maturity` field. In public
packages without private qualification evidence, the effective status is
Experimental with `qualification_evidence_unavailable`. This is a limit on
available proof, not a claim that every profile fails its simulated outcomes.

## Selecting a profile

```yaml
objects:
  chaser:
    flight_software:
      profile: fsw.profile.rpo_formation_hold.v1
      params:
        reference_object_id: target
        target_relative_state_ric_m: [0.0, 500.0, 0.0, 0.0, 0.0, 0.0]
```

Validation resolves this to the underlying stack, declared hardware, cadence,
and default parameters. Explicit `params` override profile defaults. A caller
may select another hardware profile or cadence only when the profile declares
that hardware family compatible. Normalized configuration retains both the
profile identity and resolved stack identity so review evidence can distinguish
the use-case product from its implementation engine.

Use the maintained demo as a starting point:

```bash
.venv/bin/python run_simulation.py --config configs/fsw_profile_catalog_demo.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/fsw_profile_catalog_demo.yaml
```

## Initial catalog

| Domain | Profile | Declared maturity | Groundwork provided |
| --- | --- | --- | --- |
| Baseline | `fsw.profile.coast_monitor.v1` | Experimental* | Passive typed-boundary, anomaly-monitoring, cadence, replay, and review evidence |
| Attitude | `fsw.profile.adcs_commissioning.v1` | Experimental* | Detumble, coarse-Sun recovery, Sun pointing, momentum unloading, and stack-owned FDIR |
| Attitude | `fsw.profile.adcs_nadir_payload.v1` | Experimental* | Nadir pointing with physical reaction wheels and magnetic momentum unloading |
| Attitude | `fsw.profile.adcs_sun_pointing.v1` | Experimental* | Sun-vector acquisition, loss handling, reacquisition, and physical allocation |
| Attitude | `fsw.profile.adcs_target_tracking.v1` | Experimental* | Fixed or exactly-once scheduled inertial-target tracking with loss/reacquisition |
| Orbit | `fsw.profile.orbit_maneuver_execution.v1` | Experimental* | Supplied finite-burn schedule execution |
| Orbit | `fsw.profile.leo_stationkeeping.v1` | Experimental* | Absolute ECI state tracking |
| Orbit | `fsw.profile.orbital_element_maintenance.v1` | Experimental* | Selected classical-element regulation |
| Orbit | `fsw.profile.atmospheric_pass_recovery.v1` | Experimental* | State-driven pass detection and recovery thrust |
| RPO | `fsw.profile.rpo_far_field_rendezvous.v1` | Experimental* | Planned RIC transfer, coast, correction, braking, and cleanup |
| RPO | `fsw.profile.rpo_formation_hold.v1` | Experimental* | Nonzero relative-state hold and formation groundwork |
| RPO | `fsw.profile.rpo_corridor_approach.v1` | Experimental* | V-bar approach and terminal slowdown |
| RPO | `fsw.profile.rpo_waypoint_inspection.v1` | Experimental* | Supplied RIC waypoint inspection path |
| RPO | `fsw.profile.rpo_terminal_proximity.v1` | Experimental* | Terminal closing-rate braking and settling |
| RPO | `fsw.profile.rpo_passive_retreat.v1` | Experimental* | Outward drift acquisition followed by coast |
| RPO | `fsw.profile.rpo_conjunction_response.v1` | Experimental* | Local relative keep-out response and nominal resumption groundwork |
| Low thrust | `fsw.profile.low_thrust_phasing.v1` | Experimental* | Windowed continuous-thrust relative phasing with resource gating |
| Low thrust | `fsw.profile.low_thrust_element_maintenance.v1` | Experimental* | Averaged continuous-thrust semimajor-axis and eccentricity regulation |

The catalog deliberately does not claim complete docking, operational
conjunction assessment, distributed formation consensus, flexible-body
pointing, long-arc low-thrust optimization, or certified safe-mode behavior.
Those require additional stack behavior and evidence rather than a new label.

### Declared support envelope: coast-monitor envelope

`fsw.profile.coast_monitor.v1` is supported for deterministic simulation with
`hardware.passive.v1` and task periods from 0.1 through 10 seconds. It samples
typed onboard measurements, emits monitoring diagnostics per invocation,
records missing batches and duplicate, out-of-order, stale, suspect, invalid,
and unexpected-frame measurements, and never issues actuator commands. Its
qualification covers a six-hour two-body coast, exact snapshot/replay,
three-cadence Controller Bench conformance, and a nine-run seeded orbit/cadence
campaign. The claim does not include safety action, autonomous recovery,
sensor-data repair, hardware real-time behavior, or flight certification.

### Declared support envelope: ADCS envelopes

The four exact ADCS profiles use the same rigid-spacecraft qualification
family: diagonal center-of-mass inertia of 12, 10, and 8 kg·m², three
orthogonal reaction wheels, a momentum-dump magnetorquer, 0.05-second attitude
propagation, and onboard task periods from 0.05 through 0.2 seconds. Each
profile has its own physical outcome and three-cadence Controller Bench gate;
the shared seeded campaign varies initial body rate and task cadence while
retaining exact profile identity in the evidence.

Commissioning qualifies configurable detumble hysteresis, coarse-Sun
recovery, nominal Sun pointing, wheel momentum management, and actuator
fallback. Nadir, Sun, and target profiles qualify their own moving or supplied
reference outcomes rather than inheriting commissioning maturity. Sun or
target unavailability enters a rate-damped degraded mode; clearing the typed
fault indication reacquires the reference. Target tracking also accepts
exactly-once scheduled `stack_command` inputs with `operation:
set_target_eci` and SI `target_x_eci_m`, `target_y_eci_m`, and
`target_z_eci_m` fields. Pending commands and deduplication state survive
snapshot/restart.

These claims are simulation-only. Flexible-body motion, payload jitter,
optical blinding, line-of-sight keep-outs, autonomous target acquisition,
target-motion prediction, and flight certification remain excluded.
Electrical energy or power-positive behavior is not claimed without a power
model; qualification instead bounds physical torque realization, saturation,
integrated control effort, wheel momentum, attitude error, and body rate.

### Declared support envelope: orbit-operations and RPO envelopes

The four orbit profiles share a maintained 600-second near-Earth J2-plus-drag scenario,
three seeded bounded variations, and onboard task periods from 0.25 through
1.0 seconds. Scheduled maneuver execution proves delivered impulse plus typed
receipt and physical realization, and terminal completion is withheld while
any burn command receipt remains pending or rejected. State stationkeeping and orbital-element
maintenance prove that propagated truth moves toward the supplied reference.
Atmospheric pass recovery uses onboard altitude thresholds and restart-safe
phase state; thrust is inhibited until an observed pass exit and ends after
the requested recovery delta-v. Drag-makeup policy, mean-element targeting,
general maneuver optimization, and atmospheric trajectory optimization remain
outside these exact claims.

The seven RPO profiles share a near-Earth J2 family scenario, bounded seeded
variations, and task periods from 0.1 through 0.5 seconds. Qualification spans
radial, in-track, and cross-track offsets, signed relative velocity, physical
command realization, navigation loss, actuator faults, restart, saturation,
and initially violated retreat and conjunction states. The flagship transfer
provides the deeper far-field acquire/coast/correct/brake/cleanup regression.
Surface-coverage optimization, docking/contact dynamics, distributed
formation coordination, mission-specific keep-out geometry, and operational
conjunction assessment remain excluded.

### Declared support envelope: low-thrust envelopes

The two exact low-thrust profiles are supported for deterministic simulation
inside the checked-in six-hour native ONP near-Earth envelope with J2,
exponential-atmosphere drag, ten-second orbit substeps, physical continuous
engines, and task periods from 1 through 20 seconds. The maintained scenario
uses 600-second thrust windows with 300-second active arcs. Exact command
receipts, physical force realization, propellant consumption, thrust duty
cycle, transition count, minimum dwell, gimbal feasibility, and missed-window
telemetry are machine checked.

The phasing case reduces range from 5.000 km to 1.568 km while keeping the
maximum excursion below 5.001 km, final in-track drift below 0.5 m/s, delta-v
below 0.25 m/s, and propellant use below 0.002 kg. Element maintenance uses a
configurable onboard averaging window and must finish closer to its 7000 km
semimajor-axis target than a passive comparison vehicle while containing
eccentricity, delta-v, and propellant. A three-seed campaign varies mass,
initial phase geometry, thrust-window phase, and element averaging, and the
Controller Bench independently covers 1-, 5-, and 20-second task cadences.

Typed resource measurements may inhibit thrust and the stack resumes after
power becomes available. OEL does not automatically derive availability from
eclipse, battery, thermal, or power-system models in this claim; those inputs
remain spacecraft-stack responsibilities. General low-thrust trajectory
optimization, arbitrary orbit raising, mission-specific thermal scheduling,
and flight qualification remain excluded.

## Discovery

```bash
.venv/bin/python -m sim.flight_software list
.venv/bin/python -m sim.flight_software list --kind profile --domain rpo
.venv/bin/python -m sim.flight_software show fsw.profile.rpo_far_field_rendezvous.v1 --json
.venv/bin/python -m sim.flight_software validate
```

The `materialize` command produces a normalized flight-software block when all
mission-specific parameters are supplied:

```bash
.venv/bin/python -m sim.flight_software materialize \
  fsw.profile.rpo_far_field_rendezvous.v1 \
  --params-json '{"reference_object_id":"target"}' \
  --json
```

## Qualification boundary

The public export includes profile discovery, validation, materialization, and
the declared Supported-within-envelope catalog status. Discovery reports
effective maturity separately and fails closed when qualification evidence is
unavailable. It does not include generated qualification packets, restricted
qualification inputs, or release-governance evidence, so profile discovery
remains conservative when those artifacts are unavailable. Public custom
authoring ends at content-bound validation, contract tests, and deterministic
smoke evidence; see [Public Local Edition And Private Services](public-vs-pro.md).

The local FSWDK owns profile comparison and qualification within the declared
simulation envelope. A generated qualification packet is still simulation
evidence for the exact profile and envelope, not automatic promotion or flight
qualification. Restricted inputs and release-governance evidence remain
private, and this page does not expose those private evidence paths as
reproducible public commands.

## Maturation contract

Each profile records its assumptions, known limits, required mission-specific
parameters, compatible hardware, and the common qualification gates. Promotion
requires evidence for the exact profile and version covering:

1. Closed-loop propagated-truth outcomes.
2. Sensor and navigation envelopes.
3. Accepted actuator commands and physical realization.
4. Mode transitions and stack-owned recovery.
5. Deterministic snapshot and replay.
6. Off-nominal and fault response.
7. Monte Carlo robustness.
8. Documented assumptions, limits, and tuning envelope.

The review store records `profile_id` beside `stack_id` in `fsw_invocations`.
This permits future evidence manifests and qualification tools to bind results
to the selected use-case profile without changing the sensor-to-actuator
runtime boundary.

### Mounted RCS and wheel rotor limits

For `hardware.rcs.v1`, each `params.rcs_thrusters` entry may supply
`position_body_m`: the mounting lever arm measured **from the center of mass**,
expressed in body axes in meters. `direction_body` is the normalized direction of
vehicle force, not exhaust. Native RCS hardware realizes both force and
`position_body_m × force`; omitted positions retain the zero-arm behavior.
Per-jet `isp_s` may override the shared specific impulse. Propellant-limited
realization scales the mounting torque with thrust.

Reaction-wheel hardware optionally accepts `wheel_inertia_kg_m2` and
`wheel_max_speed_rad_s` together. Both arrays must be positive and finite and
match the wheel count (a single value broadcasts). The physical momentum limit
is the smaller of the supplied momentum limit and inertia times speed limit;
telemetry includes rotor speed when inertia is configured. Existing wheel
configurations without those fields retain their previous behavior.

Translation reference stacks can use physical reaction wheels for an explicitly
configured attitude reference by selecting
`attitude_hardware_profile: hardware.reaction_wheels.v1` and supplying
`wheel_axes_body`, `wheel_max_torque_n_m`, and `wheel_max_momentum_n_m_s`.
The default coupled attitude hardware remains unchanged.
