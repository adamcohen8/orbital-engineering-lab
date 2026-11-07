# Actuators

Orbital Engineering Lab models actuator output as the force and torque actually
applied to the vehicle after actuator limits, allocation, and device dynamics.
Controllers may still emit the simple command shape:

- `thrust_eci_km_s2`: desired translational acceleration in ECI.
- `torque_body_nm`: desired body torque.
- `mode_flags`: optional actuator context and telemetry.

The public actuator package turns those desired commands into achievable
commands.

V2 force-producing hardware depletes vehicle mass using its configured
specific impulse. If a scenario gives the abstract ideal-wrench reference
profile a positive `fuel_mass_kg` but no propulsion preset or explicit
`isp_s`, OEL uses the established 220 s chemical-reference value. Concrete
RCS, continuous-engine, and electric-propulsion profiles should declare their
actual Isp (directly or through a preset).


## Public Actuator Presets

Scenarios can now use public actuator presets instead of hand-authoring the full
`specs.actuators` block. Put the preset at either `specs.actuator_preset` or
`specs.actuators.preset`; explicit fields in `specs.actuators` override the
preset.

Available presets:

- `BASIC_RCS_6DOF`: sixteen idealized body-mounted RCS thrusters with full
  six-axis force/torque allocation authority for smoke tests and controller
  bring-up.
- `BASIC_ELECTRIC_PROPULSION`: low-thrust, high-Isp electric propulsion with a
  simple power cap.
- `BASIC_MAGNETORQUER_TRIAD`: three-axis magnetorquer authority.
- `BASIC_CMG_TRIAD`: simplified three-axis CMG authority.
- `BASIC_GIMBALED_THRUSTER`: body-mounted gimbaled spacecraft thruster.

Example:

```yaml
objects:
  chaser:
    enabled: true
    kind: satellite
    specs:
      mass_kg: 250.0
      actuators:
        preset: BASIC_ELECTRIC_PROPULSION
        orbital:
          electric_propulsion:
            max_thrust_n: 0.25
```

Strict plugin validation checks preset names, actuator block shape, vector
lengths, nonnegative limits, duty-cycle ranges, RCS thruster definitions, and
fault/degradation fields before the run starts. Fields documented as
scalar-or-vector, such as magnetorquer dipole limits and reaction-wheel torque
limits, may be written as one scalar value or as a three-element vector.


## Orbital And Translation Actuators

`OrbitalActuator` supports:

- fixed body-mounted thrusters with thrust, Isp, min impulse bit, throttle-rate
  limits, lag, mass depletion, attitude coupling, and mount torque;
- RCS thruster clusters through `RcsClusterLimits` and `RcsThruster`;
- electric propulsion through `ElectricPropulsionLimits`;
- spacecraft gimbaled thrusters through `GimbaledThrusterLimits`.

RCS clusters allocate a desired body force/torque request into nonnegative
thruster firings. For force/torque allocation, the cluster geometry must provide
rank across the requested force and torque axes. The `BASIC_RCS_6DOF` preset is
regression-tested for full six-axis authority and can independently command
body-frame force and torque along X, Y, and Z within its thrust limits.

Each thruster defines:

- `position_body_m`;
- `force_direction_body`;
- `max_thrust_n`;
- optional `min_impulse_bit_n_s`;
- optional `isp_s`.

Electric propulsion is intended for low-thrust, high-Isp stationkeeping,
rendezvous, and orbit-raising studies. It limits requested thrust by max thrust,
optional power budget, duty cycle, and optional throttle response.

Gimbaled spacecraft thrusters slew the plume axis within angle and rate limits
before producing achieved thrust and mount torque.


## Attitude Actuators

`AttitudeActuator` supports:

- reaction wheels with wheel axes, torque limits, momentum limits, speed limits,
  inertia, motor lag, and friction;
- physical magnetorquers using `torque = magnetic_dipole x magnetic_field` when
  a body-frame or ECI magnetic-field vector is provided;
- pulse torque thrusters;
- simplified control moment gyros through `ControlMomentGyroLimits`;
- wheel desaturation assist through `WheelDesaturationLimits`.

When no magnetic-field vector is provided, the legacy actuator returns zero
magnetorquer torque and records no fictitious field authority.


## Fault And Degradation Layer

`FaultedActuator` and `ActuatorFaultConfig` can wrap any actuator and apply
public fault/degradation behavior:

- stuck-off actuator output;
- thrust scale factor;
- torque scale factor;
- thrust bias;
- torque bias.

This is useful for Monte Carlo, resilience testing, controller validation, and
failure-mode demonstrations.


## Telemetry

For GNC v2 ideal-wrench hardware, force and torque are always clipped to their
configured norm limits. The `saturated` telemetry flag excludes only norm excess
within eight machine epsilons relative to the limit, accounting for floating-point
rounding after controller normalization. This changes classification, not the
physical clipping calculation. There is no absolute allowance at a zero limit.
Larger physical saturation still feeds the configured health monitor and can
trigger isolation; repeated command rejections and explicit faults retain their
existing handling. Controller-side command limiting remains separate allocation
evidence and does not by itself indicate a failed actuator.

Actuator models add diagnostics to `mode_flags`, including RCS thruster forces,
electric propulsion thrust, gimbal direction/angle, reaction-wheel speeds and
momentum, magnetorquer dipole, CMG torque caps, desaturation torque, propellant
use, and fault scale factors.

Single-run outputs summarize these diagnostics under
`summary.actuator_diagnostics_summary`. Monte Carlo aggregates include
`aggregate_stats.actuator_diagnostics_by_object`, and AI-report review packets
include those actuator diagnostics in control/resource data sources.


## Scenario Configuration

Satellite objects may opt into the runtime actuator stack with
`specs.actuators`. Existing scenarios that omit this block keep the legacy
net-force/net-torque behavior.

Example:

```yaml
objects:
  chaser:
    enabled: true
    kind: satellite
    specs:
      mass_kg: 500.0
      actuators:
        enabled: true
        orbital:
          electric_propulsion:
            max_thrust_n: 0.5
            isp_s: 1600.0
            max_power_w: 100.0
            power_per_newton_w: 200.0
          gimbaled_thruster:
            neutral_direction_body: [-1.0, 0.0, 0.0]
            max_gimbal_angle_deg: 5.0
            max_gimbal_rate_deg_s: 1.0
        attitude:
          magnetorquers:
            # Scalar values expand to equal authority on all three axes.
            max_dipole_a_m2: 10.0
          control_moment_gyros:
            max_torque_nm: [0.2, 0.2, 0.2]
            momentum_nms: [1.0, 1.0, 1.0]
            gimbal_rate_limit_rad_s: [0.1, 0.1, 0.1]
        faults:
          thrust_scale: 0.95
```


## Actuator-Aware Controllers

The public control package includes starter controllers for the actuator
families above. They are deliberately conservative architecture pieces rather
than tuned mission designs:

- `sim.control.attitude.MagnetorquerBdotController`: B-field-aware detumble
  torque for magnetorquers.
- `sim.control.attitude.WheelDesaturationController`: unload torque request
  from body-frame wheel momentum.
- `sim.control.attitude.CMGSteeringController`: wrapper that caps a base
  attitude controller by simplified CMG momentum/rate authority.
- `sim.control.orbit.RCSAllocationAwareController`: wrapper that previews a
  desired force/torque against a configured RCS cluster.
- `sim.control.orbit.ElectricPropulsionController`: wrapper that caps a base
  orbit controller by low-thrust, duty-cycle, and power authority.
- `sim.control.orbit.GimbaledThrusterController`: wrapper that suppresses
  thrust directions outside a configured gimbal cone.

These are component-bench APIs. They emit the pre-v2 `Command` shape and are
not silently inserted into the v2 satellite runtime. Complete v2 stacks own
control and allocation and publish typed device commands. Use the Controller
Bench to exercise or adapt these components; use a `flight_software` stack and
compatible `hardware_profile` for a physical scenario.

Physical v2 actuator smokes:

- `configs/actuator_lab_presets_smoke.yaml`
- `configs/controller_rcs_allocation_smoke.yaml`
- `configs/controller_electric_propulsion_smoke.yaml`
- `configs/controller_gimbaled_thruster_smoke.yaml`

The magnetorquer B-dot, wheel-desaturation, and CMG controller fixtures remain
component-level checks until physical v2 hardware profiles and their telemetry
feedback contracts are promoted. Their scenario files validate hardware
metadata and the attitude reference stack, but do not claim physical execution
of those three component controllers.


## Optional Rust pipeline for built-in v2 stacks

This experimental parallel backend requires `oel-rust-orbit >=0.12.0` and
a compatible `oel-rust-game` wheel in the simulation's Python environment.
Version 0.5.0 enables batched validation; older compatible packet wheels retain
per-record native validation. Runtime-built stacks default to Rust. Select a
backend per object under the maintained flight-software params:

```yaml
objects:
  satellite:
    flight_software:
      stack: fsw.attitude_reference
      hardware_profile: hardware.reaction_wheels.v1
      params:
        numeric_backend: rust
```

The selector is also accepted by `fsw.passive`, `fsw.orbit_reference`,
`fsw.rpo_reference` and `fsw.low_thrust_reference`. Orbit and attitude dynamics
have their own independent `numeric_backend` selectors. Explicit Rust
selection requires compatible optional wheels; missing native classes raise
an error before execution.

Rust combines quaternion PD control and attitude allocation within one
controller release. Supported allocation models are ideal wrench, reaction
wheels, magnetorquers and the existing simplified CMG triad. Within each
hardware interval, Rust computes ideal-wrench lag/limits, reaction-wheel
momentum/torque limits, magnetorquer dipole/torque, simplified CMG gimbal state,
and continuous-engine thrust/mass flow. NumPy reductions and prepared matrix
products retain the reference arithmetic to preserve trajectories. Custom
attitude controllers continue through their Python control method, followed
by the supported native allocator.

Python retains release scheduling, command validity and sequence policy,
fault/availability handling, mass depletion, record construction and stored
state. Unsupported hardware, including RCS pulse/on-off and aerodynamic
position devices, retains its existing Python implementation. The selector
does not turn legacy actuator presets into concrete v2 hardware profiles.

Packet validation and internal evidence conversion are batched through the
native packet wheel. Boundary checks remain mandatory; external objects keep
the Python validator's reflection and diagnostics. The existing Python JSON
writer and public serializer still own the persisted format. Hardware state
remains available to snapshots and restore. Native checkpoint identities bind
the selected backend and implementation; cross-backend restore is rejected.
Speed and deterministic parity do not establish actuator model accuracy or
flight qualification.
