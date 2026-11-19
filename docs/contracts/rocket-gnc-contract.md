# Rocket GNC Contract

This document defines the current stabilization contract for rocket ascent
guidance, navigation, control, actuation, dynamics, and telemetry in Orbital
Engineering Lab. It is intentionally narrower than a full launch-vehicle
design specification. The goal is to make rocket behavior deterministic,
reviewable, and testable enough to mature alongside the rest of the project.

The rocket subsystem is not flight software and is not an operational launch
decision system. It is a research, training, and engineering-analysis model.


## Stability Level

This is a 0.1 rocket contract. Stable enough to rely on:

- `RocketSimConfig`, `RocketVehicleConfig`, `RocketState`, `GuidanceCommand`,
  and `RocketSimResult` as the core internal rocket data model.
- Deterministic fixed-step propagation through `RocketAscentSimulator.step`.
- `RocketAscentSimulator.run` returning histories whose final sample matches
  early termination or insertion times.
- Stage propellant depletion limiting the impulse applied over the current
  step.
- TVC-aware thrust acceleration telemetry in the single-run engine.
- `RocketNavState` as the structured derived-state packet for rocket guidance,
  telemetry, diagnostics, and tests.
- Rocket insertion summary fields in single-run payloads.

Still maturing:

- Guidance law quality and target-orbit robustness.
- Explicit navigation/filter interfaces.
- Attitude-control and TVC-control separation.
- Rocket benchmark and validation envelopes.
- Higher-fidelity staging, engine transient, and aero/structural modeling.


## Core Objects

Rocket runtime objects are created from scenario objects whose `kind` is
`rocket`. Conventional object ID `rocket` remains supported, but the engine
should treat rocket IDs as named objects rather than hard-coded global slots
where practical.

The rocket runtime state is represented by `RocketState`:

- `t_s`: simulation time in seconds.
- `position_eci_km`: inertial position in kilometers.
- `velocity_eci_km_s`: inertial velocity in kilometers per second.
- `attitude_quat_bn`: scalar-first inertial-to-body attitude quaternion. Its
  DCM maps ECI vectors into body axes (`v_body = C_bn v_eci`), matching the
  shared OEL attitude convention.
- `angular_rate_body_rad_s`: body angular rate in radians per second.
- `mass_kg`: current vehicle mass.
- `active_stage_index`: zero-based active stage index. Values greater than or
  equal to the number of stages mean all stages are spent/separated.
- `stage_prop_remaining_kg`: propellant remaining by stage.
- `payload_attached`: whether the payload is still attached to the stack.
- `thrust_vector_body`: achieved thrust-vector direction in body axes.

Derived navigation and targeting state is represented by `RocketNavState`.
It includes geodetic altitude/latitude/longitude, inertial speed components,
flight-path angle, apogee/perigee altitude, eccentricity, dynamic pressure,
Mach, aero angles, thrust-to-weight, propellant remaining, and achieved thrust
axis. Guidance implementations should prefer this packet over ad hoc duplicate
state derivations.


## Command Semantics

Rocket guidance emits `GuidanceCommand`:

- `throttle`: commanded throttle fraction. The engine clamps this to `[0, 1]`.
- `attitude_quat_bn_cmd`: desired scalar-first inertial-to-body attitude quaternion, optional.
- `torque_body_nm_cmd`: direct body torque command, optional.
- `thrust_vector_body_cmd`: desired TVC thrust-vector direction in body axes,
  optional.

When `thrust_vector_body_cmd` is absent, the rocket engine steers thrust toward
`RocketVehicleConfig.thrust_axis_body`. When present, TVC dynamics limit the
achieved vector by gimbal angle, rate, and time constant.

Mission-execution throttle overrides must preserve all non-throttle command
fields, including TVC vector commands.


## Step Order

For each outer rocket step:

1. Build mission decision context from belief-derived own state and
   observer-owned knowledge.
2. Run mission modules, mission strategy, and mission execution.
3. If launch is not authorized, hold the rocket on the pad with zero thrust
   and advance rocket time.
4. Run rocket guidance and apply mission throttle overrides.
5. Clamp throttle and propagate through `RocketAscentSimulator.step`.
6. Resolve stage propellant consumption and stage separation.
7. Apply TVC dynamics to obtain achieved thrust direction.
8. Propagate translational dynamics with gravity, optional perturbations,
   thrust, and optional aero loads.
9. Propagate attitude dynamics or apply cheater attitude mode.
10. Update truth, belief, thrust history, rocket metrics, and termination state.

The rocket path should follow the broader engine contract: agent-facing
decisions should not depend on raw hidden world truth.


## Staging And Engine Impulse

Within a step, stage propellant consumption is bounded by propellant remaining.
If a stage runs out before the end of the step, the engine applies only the
impulse-equivalent average thrust for the burned portion of that step. The
stage dry mass is separated after propellant depletion.

This is still a fixed-step approximation. Future higher-fidelity work may split
the step at burnout time, but it must preserve this contract-level rule:
depleted propellant may not produce a full-step thrust impulse.


## Termination And Insertion

Rocket ascent may terminate early for:

- `earth_impact`,
- `rocket_landing` or `rocket_crash` for opt-in first-contact classification
  ([scenario configuration](../scenario-yaml.md#rocket-touchdown-classification)),
- `rocket_orbit_insertion`,
- future explicit rocket failure reasons.

Earth-impact termination uses the WGS-84 ellipsoidal surface when
`use_wgs84_geodesy` is enabled and `earth_impact_radius_km` has its default
value of 6378.137 km (the WGS-84 equatorial semi-major axis). Setting a
different `earth_impact_radius_km` selects a spherical impact boundary at that
radius, while WGS-84 geodesy remains active for navigation and altitude
reporting. With `use_wgs84_geodesy` disabled, impact termination always uses
the configured spherical radius.

For early termination, returned result histories must include the post-step
state at `termination_time_s`. Time histories must be strictly representative:
the final sample time, truth state, mass, stage index, and orbital elements
must describe the final returned state rather than uninitialized array storage.
For a configured touchdown, the final rocket state is repropagated to the
bracketed contact time with the already-issued command. The touchdown record
reports the pre-contact state, Earth-fixed velocity, and configured limit
results.

With [deployable landing gear](../scenario-yaml.md#deployable-rocket-landing-gear),
the rocket publishes the first footpad or body contact separately from
settling or crash. The guidance and mission execution interfaces carry
per-leg deploy or stow requests. The simulated actuators own the achieved
extension and lock state, and the dynamics own contact forces and torques.

Insertion is achieved when:

- altitude is within `target_altitude_tolerance_km` of `target_altitude_km`,
- eccentricity is less than or equal to `target_eccentricity_max`,
- all vehicle stages are complete,
- the condition holds for `insertion_hold_time_s`.


## Telemetry

Rocket single-run outputs should report:

- throttle command,
- impulse-equivalent thrust,
- stage index,
- dynamic pressure,
- Mach number,
- wind in body axes,
- TVC gimbal angle,
- angle of attack and sideslip,
- aerodynamic force and moment magnitudes,
- insertion achieved/time,
- termination reason/time/object.
- summary fields under `summary.rocket_metrics_summary`.

Applied thrust history must use the actual achieved thrust vector, including
TVC deflection, not only the nominal vehicle thrust axis.

Guidance phase codes currently use:

- `1`: ascent,
- `2`: coast to apogee,
- `3`: circularize,
- `4`: complete.

Unknown or unavailable phase values are recorded as `NaN`.


## Regression Expectations

Focused tests should protect:

- early termination histories include the post-step state,
- propellant depletion limits average step thrust,
- stage index and propellant histories advance consistently,
- TVC vector commands survive throttle overrides,
- applied rocket thrust telemetry follows the achieved TVC vector,
- guidance wrapper ordering is preserved,
- max-Q limiter behavior accounts for wind-relative velocity.

Broader rocket maturity work should add benchmark scenarios for nominal ascent,
stage separation, TVC tracking, max-Q limiting, wind stress, payload margin, and
closed-loop insertion performance.
