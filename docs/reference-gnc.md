# GNC v2 Reference Flight-Software Stacks

OEL satellites use one `SatelliteFlightSoftware` boundary. Typed sensor and
input events enter; typed device-coordinate actuator commands leave. The stack
owns navigation, belief, goals, constraints, executive decisions, guidance,
control, allocation, task scheduling, and its own recovery procedures.

All included v2 stacks are currently **Experimental**. Their evidence is
simulation evidence, not flight qualification, operational safety approval, or
a claim outside the tested envelope.

## Selection

```yaml
objects:
  chaser:
    kind: satellite
    initial_state:
      relative_to: target
      relative_ric_rect: [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    flight_software:
      stack: fsw.rpo_reference
      task_period_s: 0.1
      hardware_profile: hardware.ideal_wrench.v1
      params:
        reference_object_id: target
        translation_mode: v_bar_approach
        max_acceleration_m_s2: 0.001
```

A passive satellite may omit the section; normalization selects `fsw.passive`.
For a dynamics-only object that does not need an onboard boundary, set
`runtime_profile: trajectory_only` and omit both `flight_software` and
object-owned `knowledge`. This explicit profile retains deterministic
propagation while producing no FSW invocations or telemetry.
Any non-empty v1 satellite controller, mission, or bridge field fails with
migration guidance. A custom stack uses `module` plus `class_name`, and receives
only the declared `params`.

## Included stacks

| Stack | Purpose | Current physical profile |
| --- | --- | --- |
| `fsw.passive` | Coast/no-command baseline with typed boundary evidence | Passive |
| `fsw.attitude_reference` | Quaternion/reference generation, torque control, and allocation | Ideal torque/wrench |
| `fsw.orbit_reference` | Absolute-orbit stationkeeping and orbital-element feedback | Ideal wrench |
| `fsw.rpo_reference` | RIC hold, axis approaches, waypoint, planned RIC PD transfer/coast/correction/final cleanup, terminal braking, and passive retreat | Ideal wrench |
| `fsw.low_thrust_reference` | Low-thrust relative phasing | Continuous engine and body-frame gimbal commands |
| `fsw.game_pilot_reference` | Typed pilot/operator input profiles | Ideal wrench or modeled aerodynamic effectors |

Component implementations under `sim.gnc` have a separate catalog and maturity
record. They are composition building blocks, not runtime slots. Public users
can compose them in a custom complete stack; Pro adds Controller Bench
orchestration, not additional runtime authority.

For prebuilt application-level selections, see
[Flight-Software Use-Case Profiles](flight-software-profiles.md). These
versioned profiles resolve to the stacks above while retaining independent
per-version, envelope-bounded Supported maturity and review provenance. The
underlying stacks remain Experimental for arbitrary composition.

## Units, frames, timing, and safety

- Boundary and onboard calculations use SI. Review plots and reports may use
  engineering notation.
- Measurements identify their sensor/source frame. Device commands identify
  the actuator frame. The stack owns any internal transforms.
- Sensor sampling, scheduled stack releases, actuators, coupled dynamics, and
  output sampling may run at independent deterministic cadences. The reference
  runtime is event-driven: a due sensor sample or other releasable input invokes
  the complete stack at that boundary; it is not a separately scheduled sensor
  task inside the stack. Standard review detail records release reasons in
  `fsw_task_timing.detail_json`. Modeled duration, execution budget, deadline,
  and missed-deadline disposition are separate columns; host profiling is
  distinct from modeled onboard timing.
- The attitude substep may be smaller than the orbit substep and never larger.
  Quaternion propagation retains the exponential-map evolution inside the
  stage-consistent coupled integrator.
- The stack owns recovery behavior. OEL records configured safety requirements
  and post-run violations; it does not inject a global satellite safety policy.

## Review evidence

Enable the review store and query the v2 tables:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

```bash
.venv/bin/python -m sim.review outputs/<run> --query \
  "SELECT object_id, invocation_id, stack_id, profile_id, input_count, command_count FROM fsw_invocations"
.venv/bin/python -m sim.review outputs/<run> --query \
  "SELECT object_id, command_source_id, command_boot_id, command_sequence, disposition FROM actuator_command_receipts"
.venv/bin/python -m sim.review outputs/<run> --query \
  "SELECT object_id, objective_id, state FROM fsw_objectives"
```

Commands link to their source invocation and receipt. Physical realization links
to the accepted command identity. Onboard diagnostics remain structurally
separate from truth-derived review and scoring evidence.

To include actuator identity, join `actuator_command_receipts` to
`actuator_realization` on object ID plus command source, boot ID, and sequence;
one accepted command may have more than one realization row.

## Maintained examples

```bash
.venv/bin/python run_simulation.py --config configs/reference_gnc_vbar_approach.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/reference_gnc_nadir_pointing.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/reference_gnc_rpo_executive.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml --validate-only
.venv/bin/python run_simulation.py --config configs/controller_electric_propulsion_smoke.yaml --validate-only
```

See [flight-software profiles](flight-software-profiles.md) for the public
profile-selection and maturity workflow. The cFS/SIL adapter and its ICD are a
Pro/private integration excluded from the public export.
