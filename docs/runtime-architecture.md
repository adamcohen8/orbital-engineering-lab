# Runtime Architecture

`sim.runtime_support` remains OEL's compatibility façade for runtime records,
state initialization, object factories, mission dispatch, and command helpers.
New code should locate implementation ownership through
`RUNTIME_CONSTRUCTION_FAMILIES` in `sim.runtime.architecture`:

- `models.py`: `AgentRuntime` and rate-limited controller state
- `compat.py`: plugin construction and call-signature compatibility
- `state_initialization.py`: Cartesian, COE, TLE, CR3BP, and relative states
- `actuator_factory.py`: propulsion, actuator stacks, and mass properties
- `satellite_factory.py`: satellite dynamics, estimation, control, and runtime construction
- `rocket_factory.py`: rocket stacks, guidance, ascent configuration, and runtime construction
- `knowledge_factory.py`: tracked-object knowledge and EKF configuration
- `mission_runtime.py`: deployment, mission modules, strategy, and execution
- `commands.py`: command serialization and decision-state views

`sim.single_run._SingleRunEngine` remains the stable lifecycle coordinator.
Its focused collaborators are:

- `sim.execution.runtime_profile`: runtime timing and profile payloads
- `sim.execution.object_workers`: persistent process-worker transport
- `sim.execution.object_step_coordinator`: serial/parallel planning and executor selection
- `sim.execution.system_forces`: opt-in serial adapter for synchronized pair stepping;
  `sim.dynamics.orbit.system_force` owns the shared-stage RK4 derivative
- `sim.execution.single_run_history`: history growth, retention, and compaction
- `sim.execution.passive_history`: bounded native RK4 two-body/J2 histories for
  eligible attitude-disabled passive objects; ordinary sample, event, knowledge,
  and termination processing still consumes each row in order
- `sim.reporting.run_payload_assembly`: reporting views, payload construction, and artifact dispatch

Public-core single-scenario execution uses deterministic serial object
stepping. Process-pool object stepping is a Pro-entitled feature. `auto` may
select a compatible available backend or fall back to serial under resource or
compatibility constraints; explicitly forcing an unavailable parallel backend
fails the applicable feature/validation check.

External callers should continue using `sim.execution`, `sim.api`, or the
existing compatibility modules. These implementation modules are ownership
boundaries, not a replacement public API.


## Planetary trajectories

Non-Earth passive trajectories dispatch once at construction to `sim.dynamics.planetary.runtime`; the dedicated native `PlanetaryPlan` integrates forces without Python stage callbacks. Planetary reporting uses separate neutral state tables. Earth runtime and force ABI remain on their established path. See [Planetary ONP](planetary-onp.md).

Opt-in `simulator.additional_gravity` selects `EarthAdditionalGravityPropagator`
at factory construction or wraps the planetary native plan. Its separate Rust
composition evaluator adds complete relative external fields; ordinary Earth
force and integration loops have no additional-body dispatch. See
[extended third-body gravity](additional-gravity.md).
