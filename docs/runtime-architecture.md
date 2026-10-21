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
- `sim.execution.object_workers`: persistent process-worker transport and the
  explicit native-history thread executor, which retains serial object stepping
- `sim.execution.thread_workers`: bounded complete object-step threads with
  private persistent engine state and serialized input/result ownership
- `sim.execution.thread_contract`: exact built-in component identities and
  implementation declarations for thread-safe extensions
- `sim.execution.object_step_coordinator`: serial/parallel planning and executor selection
- `sim.execution.system_forces`: opt-in serial adapter for synchronized pair stepping;
  `sim.dynamics.orbit.system_force` owns the shared-stage RK4 derivative
- `sim.execution.single_run_history`: history growth, retention, and compaction
- `sim.execution.passive_history`: bounded native RK4 two-body/J2 histories for
  eligible attitude-disabled passive objects; ordinary sample, event, knowledge,
  and termination processing still consumes each row in order. Explicit Pro
  `thread_pool` execution prepares independent two-body/J2 histories in bounded
  groups without moving Python runtime callbacks or cache mutation to threads
- `sim.reporting.run_payload_assembly`: reporting views, payload construction, and artifact dispatch

Public-core single-scenario execution uses deterministic serial object
stepping. Process-pool object stepping is a Pro-entitled feature. `auto` may
select a compatible available backend or fall back to serial under resource or
compatibility constraints; explicitly forcing an unavailable parallel backend
fails the applicable feature/validation check.

Explicit Pro `thread_pool` selection can also execute complete supported ONP
object steps. The planner's `thread_scope` selects `object_step`,
`native_history`, or `auto` within this explicitly requested backend. The
automatic scope uses the narrow native-history optimization when every object
qualifies and the binding releases the GIL; otherwise it uses complete object
steps. This scope choice is separate from execution `policy: auto`, whose
thread crossover remains deferred.

Complete thread workers reuse the process worker's construction and step
functions. Engine snapshots, step messages and results cross an explicit
serialization boundary to create independent mutable state. Scenario time,
ordered result application, knowledge synchronization, events, termination,
and output writing remain owned by `_SingleRunEngine`. The runtime rechecks
component declarations before threaded submission. Native numerical sections
may release the GIL; Python callbacks retain ordinary interpreter execution.

Thread execution shares one address space. It cannot contain a native crash or
forcibly terminate a running thread. Cancellation and shutdown must wait for
running code to finish or notice a cooperative cancellation request. The
process backend remains available when dependency safety or failure isolation
requires a process boundary.

Independent Pro campaigns use `sim.execution.thread_campaign` when their
transport is explicitly set to `thread_pool`. This covers Monte Carlo,
sensitivity, config queues, controller benchmarks and controller optimization.
Task and result graphs are serialized, while the owner handles progress,
checkpoints, retention and aggregate evidence. `sim.execution.worker_budget`
carries hierarchical allocations through execution-local contexts rather than
mutating the environment shared by campaign threads. Inner process pools are
rejected; supported inner object threads receive the allocated budget.

On macOS 15 and later, `sim.execution.thread_math` applies Accelerate's
per-thread single-thread BLAS/LAPACK setting during worker calls and restores
it afterward. Other native math runtimes retain their startup limits; OEL's
worker cap alone does not bound a third-party library's internal threads.
Campaign threads reject per-run plotting/animations, SPICE kernel state and
unsupported extension hooks. Plotting and aggregate evidence remain owner
operations. Scale queue and campaign transports follow the same thread
ownership contract.

The native `NATIVE_OBJECT_KERNELS_GIL_RELEASED` marker describes the listed
Rust-owned mechanical, attitude, force, control and estimation kernels.
Callback-bearing integrators still hold the GIL while calling Python; the
marker does not promise that every native entry point runs concurrently.

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
