# Engine Contract

This document defines the current execution contract for Orbital Engineering Lab
single-run simulation workflows. It is intentionally narrower than the full
implementation: it describes behavior users, tests, docs, and future
integrations may rely on, and it calls out areas that are still being
consolidated.

The contract applies to:

- public CLI single-run execution,
- `SimulationSession` and public API single-run execution,
- deterministic single-run scenarios used by validation and examples.

Local Monte Carlo, sensitivity, and Controller Bench workflows reuse the
scenario model and single-run engine. Their orchestration and artifacts fall
outside this contract.


## Stability Level

This is the stable single-run contract for OEL 1.x. The
[compatibility policy](../compatibility-policy.md) governs additive changes,
deprecation, and corrections. Implementation ownership may change while
documented entry points and behavior remain compatible.

Stable enough to rely on:

- deterministic step indexing and time grid,
- single-run object lifecycle,
- top-level step ordering,
- snapshot shape and timing,
- termination summary semantics,
- public API entrypoints for loading, running, and stepping scenarios.

Still maturing:

- exact payload field stability beyond documented summary fields,
- explicitly experimental batch/benchmark extensions beyond their published
  versioned contracts.


## Canonical Entrypoints

Preferred public entrypoints:

- CLI: `.venv/bin/python run_simulation.py --config <path>`
- API: `SimulationConfig.from_yaml(...)` and `SimulationSession`
- Execution service: `sim.execution.run_simulation_config_file(...)`

For single-run scenarios, these routes should converge on the same conceptual
execution model. A user should not need to understand private orchestration
paths to reason about normal single-run behavior.

Private or transitional internals:

- `sim.master_simulator` is a compatibility facade for legacy imports.
- Batch analysis and campaign workflows are owned by `sim.execution`.
- Single-run payload construction and artifact writing are split between the
  engine-facing payload builder and reporting/artifact writers. The engine is
  responsible for simulation state evolution; reporting modules consume the
  resulting payload.
- Internal classes prefixed with `_`, including `_SingleRunEngine`, are not
  public extension APIs even when documented here for behavior.
- The legacy lower-level `SimulationKernel` loop has been removed; single-run
  behavior should be reasoned about through the canonical engine contract here.


## Time Grid

A single run uses a fixed outer time step:

- `dt_s` is the outer simulation step.
- `duration_s` defines the nominal run duration.
- The nominal time history includes the initial sample at `t = 0`.
- For standard single-run execution, the intended sample count is
  `floor(duration_s / dt_s) + 1`.

Substepping:

- Orbit and attitude propagation may use `orbit_substep_s` and
  `attitude_substep_s`.
- The effective internal satellite substep is the smaller of the active orbit
  and attitude substeps.
- Orbital command sampling may be slower than attitude/dynamics substeps through
  `orbit_command_period_s`.

Config validation should reject timing grids that cannot be represented cleanly.
Scenario authors should treat non-divisible timing combinations as invalid
unless an explicit migration or compatibility note says otherwise.


## Object Lifecycle

Supported object model:

- Scenario YAML should define active scene participants under the canonical
  `objects` map.
- Object IDs are user-facing names and may be domain-specific, such as
  `inspection_sat`, `depot`, `chief`, or `deputy`.
- Conventional IDs `rocket`, `chaser`, and `target` remain supported as names
  and compatibility aliases, but they are not fixed engine slots.
- Object `kind` selects the runtime family, currently `satellite` or `rocket`.

Object creation:

- Enabled config sections create runtime objects.
- Disabled config sections do not participate in truth, belief, control,
  sensing, or output histories.
- Relative initialization, knowledge targets, controller objectives, benchmark
  objectives, and output histories refer to object IDs by name.

Initial sample:

- At `t = 0`, active objects write initial truth history.
- Initial belief history is written when a belief state exists.
- Knowledge bases, when configured, write an initial snapshot for known targets.

Activation:

- Most enabled objects are active at initialization.
- A satellite configured for rocket deployment may exist but remain inactive
  until deployment time.
- Inactive objects are skipped for propagation/control until activated.

Rocket-specific lifecycle:

- Rocket guidance and rocket simulation are handled by the rocket runtime path.
- Rocket waiting-for-launch state may hold the vehicle without thrusting.
- Rocket insertion can terminate `rocket_ascent` scenarios when configured
  insertion criteria are satisfied.


## Step Order

For each outer step from `t_k` to `t_{k+1}`, single-run execution follows this
conceptual order.

1. Resolve time and environment context.
2. Activate any time-gated objects, such as rocket-deployed chasers.
3. Build an internal current-time world-truth snapshot from active objects at
   `t_k`.
4. Propagate optional reference trajectories, such as target reference orbit.
5. For each active object, execute its runtime path:
   - rocket path: mission modules, mission strategy, mission execution,
     guidance, rocket propagation, belief update, thrust/mass/stage metrics;
   - satellite path: the v2 runtime releases due typed sensor and external
     input packets to the complete flight-software stack, whose navigator,
     executive, guidance, control and allocator produce typed device commands;
     the adapter validates and realizes them before physical propagation.
     Explicit `trajectory_only` satellites bypass onboard logic. Non-empty
     retired satellite controller/mission fields are rejected.
6. Update bridges for objects with enabled bridge integrations.
7. Update object knowledge bases from the post-step world truth.
8. Write truth, belief, knowledge, applied thrust, applied torque, desired
   attitude, and runtime/debug histories for `t_{k+1}`.
9. Emit the step callback, if one is registered.
10. Evaluate termination conditions.
11. Return a snapshot for the current index when stepping interactively.

The decision boundary contains typed observable packets, never raw world-truth
objects. The runtime owns sensor generation and applies configured range,
FOV, Earth line-of-sight and dropout conditions before target packet delivery.
The stack owns navigation propagation, measurement acceptance and command
selection. Missing observations therefore remain missing, including when
navigation initialization is `ideal` but access conditions are explicit.

The separate observer knowledge base is a reporting/analysis path, not the
source of the v2 stack's navigation solution. Both paths use the same access
policy implementation; their schedules and stochastic histories are separate.
See [Flight-software observations](../flight-software-observations.md).

## Truth, Belief, And Knowledge Timing

Truth:

- Truth at index `0` is the initial condition.
- Truth at index `k + 1` represents propagated state at `t_{k+1}`.
- Satellite decision logic receives typed sensor/input packets and constructs
  its own navigation state. It cannot fall back to raw simulator truth.
- The selected stack determines how stale or missing target estimates affect
  control. Loading an initial state or using ideal measurements is explicit.
- Dynamics and the runtime adapter may access physical truth for propagation,
  sensor generation and device realization.

Belief:

- Satellite API belief is the latest navigation state published by the stack
  through `oel.navigation_state.v1` telemetry. It is not the latest sensor
  measurement and the adapter never advances a filter just to report a value.
- Telemetry generation time identifies the invocation. State epoch and age
  distinguish held state from a propagated estimate. Outer samples retain the
  latest published state; inspect these times when task cadence is slower.
- Raw measurements remain distinct in `fsw_input_events`; estimated state and
  its metadata are available in `fsw_diagnostic_fields`.
- A stack that publishes no navigation state has unavailable belief (NaN components, or an empty vector before any state shape is known).
  Missing components are not filled from truth or raw measurements.
- Orbit vectors in API history retain km and km/s; boundary telemetry uses SI.
  Quaternion/rate components, when published, retain their existing ordering.

Knowledge:

- Knowledge bases are observer-owned and update after the outer propagation
  step. Their estimator is selected by `knowledge.estimation`.
- The complete stack's orbit filter is selected independently by
  `flight_software.params.navigation_filter`.
- These are separate estimates. A reporting detection statistic is not proof
  that a particular packet entered the stack; inspect the typed packet stream.

Truth is physical simulation state; belief is published onboard navigation;
knowledge is a separate observer track. Replay verifies reproducibility of
these quantities, not their physical accuracy.

## Controller And Actuator Timing

Satellite control:

- Orbit and attitude controllers may be evaluated during internal substeps.
- Stack controllers act on their own navigation solution corresponding to the
  start of the interval they command.
- Mission modules, mission strategy, external intent providers, and mission
  execution can modify or replace controller commands.
- Integrated mission commands may bypass separate orbit/attitude command
  combination when `mission_use_integrated_command` is set.
- Orbital thrust commands may be latched and reused until the orbital command
  period elapses.

Actuator limiting:

- Commands are constrained by available mass, dry mass, max thrust, Isp-derived
  propellant use, attitude-coupled thruster direction, and torque logic when
  configured.
- Applied thrust and torque histories represent the command actually applied to
  dynamics, not merely the raw controller request.

Runtime budgets:

- The runtime records host execution duration separately from modeled task
  timing. It is not a hard realtime scheduler and does not revive the removed
  kernel's overrun-command substitution behavior.


## Snapshots

`SimulationSession.step()` and engine snapshots expose:

- `step_index`
- `time_s`
- `truth`
- `belief`
- `applied_thrust`
- `applied_torque`

Snapshot semantics:

- A reset single-run session returns a snapshot at index `0`.
- A step advances at most one outer time index unless the run is already done.
- Calling step after completion returns the final snapshot.
- Snapshots are only available for single-run scenarios, not batch analysis.


## Termination

Nominal termination:

- A run completes when the current index reaches the final time-grid index.

Early termination:

- Earth-impact termination can stop active object scenarios.
- Rocket insertion can stop `rocket_ascent` scenarios when insertion criteria
  are met.

Termination metadata:

- `terminated_early`
- `termination_reason`
- `termination_time_s`
- `termination_object_id`

Downstream tools should use these fields instead of inferring early termination
only from sample count.

Opt-in `simulator.collisions` does not terminate a run. For the supported
two-satellite passive ONP envelope, the engine resolves each first spherical
contact within the step, applies a frictionless restitution-1 impulse to both
ECI velocities, then advances the remaining interval. Ordinary state samples
remain on the time grid. Exact impact times and pre/post velocities are recorded
in `collision_events` and the review `events` table.


## Payload And Artifact Expectations

Single-run payloads should include:

- time history,
- truth histories by object,
- belief histories by object,
- knowledge histories by observer/target when configured,
- applied thrust histories,
- applied torque histories,
- desired attitude histories when configured,
- controller debug history where available,
- rocket metrics when a rocket is present,
- summary metadata,
- plot and animation artifact paths when enabled.

The summary is a stable review surface for OEL 1.x. It should include:

- scenario name and description,
- object IDs,
- sample count,
- `dt_s` and duration,
- termination status,
- thrust statistics,
- attitude guardrail statistics,
- plot and animation output manifests.

Detailed payload arrays and debug structures are useful but still maturing.
If a downstream tool depends on a detailed field, add or update tests that
codify that dependency.


## Public Extension Points

Supported extension surfaces:

- scenario YAML configuration,
- object presets,
- plugin pointers validated by `validate_scenario_plugins`,
- controller/mission classes that implement the expected methods,
- `SimulationSession` and public API wrappers,
- external intent providers registered on a `SimulationSession`.

Internal surfaces:

- private helper functions,
- `_SingleRunEngine` implementation details,
- exact controller debug record contents,
- legacy master-simulator orchestration,
- undocumented batch-analysis implementation details.

When in doubt, prefer extending through config, presets, validated plugin
pointers, or the public API rather than importing private helpers.


## Compatibility Rules

Changes should update this document when they alter:

- step order,
- object activation semantics,
- truth/belief/knowledge timing,
- controller or actuator timing,
- termination semantics,
- stable summary fields,
- public API behavior,
- supported extension points.

Preserve supported behavior throughout 1.x under the compatibility policy.
Document numerical and security corrections with focused tests and release
notes; version incompatible successors explicitly.


## Known Gaps

- Experimental batch/benchmark extensions have only their declared maturity;
  use the owning workflow contract for its supported envelope.
- Generated release qualification packets remain private evidence. Public
  validation recipes and inputs support only their documented bounded claims.
