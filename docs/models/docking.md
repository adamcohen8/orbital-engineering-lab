# Cooperative docking dynamics

OEL can simulate two cooperative rigid spacecraft through port contact, soft
capture, and joined flight. The model is experimental and uses a generic
ring-and-face interface with illustrative stiffness, damping and latch
parameters. Predicting loads or a capture envelope for specific hardware
requires measured interface data and comparison against independent test
results.

## Run the examples

From the source checkout with the documented Python environment active, choose
`capture`, `rebound`, or `mate`.

```bash
case_name=mate
config="examples/configs/public_docking_${case_name}.yaml"
python run_simulation.py --config "$config" --validate-only
python run_simulation.py --config "$config"
```

| Example | Initial geometry | Expected events |
| --- | --- | --- |
| [Soft capture](../../examples/configs/public_docking_capture.yaml) | Centered approach with opposed ports | First contact, soft capture |
| [Rebound](../../examples/configs/public_docking_rebound.yaml) | Tilted, laterally offset approach outside the latch envelope | First contact, rebound |
| [Hard mate](../../examples/configs/public_docking_mate.yaml) | Centered approach with hard-mate criteria enabled | First contact, soft capture, hard mate |

## Two-vehicle flight-software example

[The closed-loop scenario](../../examples/configs/public_docking_flight_software.yaml)
starts with the ports about 35 cm apart along the approach axis and a
0.1 rad attitude error. Both vehicles run the same example stack class
with different roles. The chaser aligns, checks the receiver's arm
indication, and closes. The receiver holds inertial attitude and coasts
in translation. Both stacks receive ideal port-relative pose, velocity,
and mechanism indications at 20 Hz. They separately arm soft capture and,
after sensing capture, request hard mate.

```bash
config=examples/configs/public_docking_flight_software.yaml
python run_simulation.py --config "$config" --safe-validate
python run_simulation.py --config "$config" --validate-only
python run_simulation.py --config "$config"
```

Review `fsw_diagnostic_fields` for each vehicle's mode, `actuator_commands`
and `actuator_command_receipts` for command acceptance, and
`actuator_realization` and `actuator_device_state` for applied wrenches and
mechanism requests. `docking.json` records the paired capture and hard-mate
authorizations alongside the physical events.

```bash
run_dir=outputs/examples/public_docking_flight_software
python -m sim.review "$run_dir" --query \
  "SELECT object_id, value_text AS mode, MIN(generated_time_ns)/1e9 AS first_time_s
   FROM fsw_diagnostic_fields
   WHERE topic='oel.example.docking.v1' AND field_name='mode'
   GROUP BY object_id, value_text ORDER BY first_time_s, object_id"
```

The example begins at final approach and uses ideal sensing, fixed-mass
ideal wrench devices, and an ideal peer arm indication. It has no far-field
guidance, sensor calibration, communications link model, or checkpoint
support.

## Scenario configuration

Enable `simulator.docking` for two enabled satellites with explicit Cartesian
ECI initial states and initially separated, opposed ports. The passive examples use
`trajectory_only`. With `commanded: true`, both satellites use custom
flight-software stacks with fixed-mass ideal wrench hardware. The model
requires both current capture-arm commands at contact and both hard-mate
requests throughout the configured lock dwell. Commands expire through the
actuator bus. The physical capture and lock envelopes still apply. Give both spacecraft their mass and center-of-mass inertia.
The paired integrator supports attitude-enabled, serial, Earth-centered ONP
with central gravity and RK4. It uses the Python stage derivatives. Bridges,
alternate force models, attitude disturbances, spacecraft resources, and
individual API impulses are outside the supported envelope.

Each `probe` and `receiver` has an `object_id`, an `offset_body_m` from
the spacecraft center of mass, a `radius_m`, and optionally a scalar-first
unit `quat_port_from_body` mapping body coordinates to port coordinates.
The identity mapping is the default. Both ports use local +Z as their
outward normal. At ideal alignment the port X axes agree, and the Y and Z
axes oppose each other. The receiver radius must cover the centered probe ring.

- `max_step_s` bounds the synchronized physics substep. Smaller orbit or
  attitude substep limits also apply. Use a step that resolves contact
  time, port motion near the edge of the receiver, and force peaks.
- Four contact points are distributed equally around the probe ring.
  A point inside the receiver face can press against it. With signed point
  gap (g) and gap rate `g_dot`, its normal load is
  `max(0, -contact_stiffness_n_m * g - contact_damping_n_s_m * g_dot)`
  when the faces oppose and the point reaches the receiver plane. Stiffness
  and damping are per point. Both spacecraft receive the same contact force
  with opposite signs, and each torque uses the lever arm from its own
  center of mass. `max_compression_m` bounds model travel and raises a
  runtime error if exceeded at a checked substep.
- The optional `latch` block specifies the lateral, angular, speed and
  rate limits for engagement at onset of a contact episode. Active latch
  springs and dampers act on relative port pose and motion. The latch
  releases if the connection exceeds configured travel, force, or torque.
  A released latch cannot re-engage in the same run.
- The optional `hard_mate` block requires a latch. Its position, angle,
  speed and rate tolerances must hold for a simulated dwell period. The
  lock retains the as-latched relative pose, combines inertias via the
  parallel-axis theorem and projects velocities to rigid motion while
  conserving total linear and angular momentum. The joined pair advances
  with fixed relative position and attitude and both object IDs remain in
  the state history. The lock is permanent for the remainder of the run.

The four frictionless contact sites approximate a ring against a flat disk.
The model covers local approaches to the mating faces. The interface
does not represent hull collisions, detailed latch geometry, friction,
flexible structures, or hardware damage. Contact onset is refined within a
bracket found at the endpoint or midpoint of a physics substep. Rebound,
latch release, and lock conditions are checked at physics substeps. Verify
convergence for shortened contact intervals and edge passages.

## Inspect the run

`docking.json` contains the final phase, peak sampled interface loads, event
states, and pose and load histories even with the full run log disabled.
Contact and soft-capture phases add substep rows to the ordinary scenario
samples. The load fields describe contact and latch forces. Constraint
reaction loads in the hard-mated phase are outside this model and load
fields are zero in that phase. The summary separately lists the joined
mass and inertia in the receiver body axes.

```bash
run_dir=outputs/examples/cooperative_docking_mate
python -m sim.review "$run_dir" --query \
  "SELECT time_s, event_type FROM events WHERE source = 'docking_dynamics' ORDER BY time_s, event_id"
python -m sim.review "$run_dir" --query \
  "SELECT time_s, phase, gap_m, interface_force_n, body_torque_n_m FROM docking_samples ORDER BY interface_force_n DESC LIMIT 8"
```

The numerical tests compare the centered impact to the damped and undamped
reduced-mass spring solutions, check momentum through off-axis contact and
the hard lock, and compare event times and loads at two maximum step sizes.
For the engineering basis of spring contact and comparison against
mechanism data, see the [NASA two-body contact model](https://ntrs.nasa.gov/citations/19870016368)
and [NASA docking-mechanism test study](https://ntrs.nasa.gov/citations/19930012222).

## External comparisons

### Basilisk welded spacecraft comparison

Run with a Python environment containing OEL's dependencies and a
MuJoCo-enabled Basilisk installation. Use a new output directory each time.

```bash
python -m examples.docking_validation.basilisk_docking_reference \
  --output-dir outputs/validation/docking_basilisk
```

The runner adapts [Basilisk's docking example](https://avslab.github.io/basilisk/examples/mujoco/scenarioSimpleDocking.html)
with shared initial conditions and disabled contact. It compares Basilisk's
site weld against the OEL assembly used after hard mate for torque-free
rotation, an off-axis assembly load from one spacecraft, and weld activation
with residual relative motion. The activation comparison starts after a
0.25 s settling interval, with the full transient retained in the histories.
Both engines run at 2 ms and 1 ms. OEL's load case injects force at one
constituent's center of mass through the assembly acceleration input at
the model level. The standard scenario flight-software path is outside
that comparison.

The output includes the MuJoCo scene, both sampled state histories, source
hashes, build identity, tolerances, and maximum differences. The comparison
covers ideal rigid joined motion in free space. Contact forces, latches,
external orbital forces, weld reaction loads, and flight qualification
remain outside its scope.

### Tohoku air-bearing contact correlation

Download the [arXiv v1 paper](https://arxiv.org/pdf/1309.3512v1) and use a Python
environment with OEL dependencies and `pypdf`.

```bash
python -m examples.docking_validation.tohoku_docking_reference \
  --reference-pdf /path/to/1309.3512.pdf \
  --output-dir outputs/validation/docking_tohoku
```

This bench harness calls OEL's paired stepper and contact law directly. It
uses a prescribed constant-speed target, a 63.2 kg freely moving body, and
the reported total stiffness and damping distributed across four symmetric
sites. The authors estimated damping from this experiment. The harness uses
the published value without refitting. It digitizes the PDF's vector traces
from Figure 13(b) and compares at 2, 1, and 0.5 ms. The digitized force and
velocity, simulated traces, overlay, and numerical comparison are saved
alongside source hashes. The force threshold is an analysis convention for
a noisy trace. The report also records sensitivity to that convention.

At 0.5 ms, OEL produces 4.45 N peak force versus the reported 5.0 N
(5.21 N on the digitized trace), 15.11 mm/s rebound versus 15 mm/s,
2.219 N·s impulse versus 2.227 N·s integrated from the figure, and 0.713 s
above the 0.5 N contact threshold versus 0.707 s from the figure. Threshold
duration spans the first to last crossing. The raw instrument telemetry was
unavailable. These results correlate axial rebound and impulse for one
bench case with fitted damping. Peak-load fidelity, off-axis contact, and
soft capture require separate evidence.
