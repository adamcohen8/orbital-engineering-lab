# Experimental impulsive trajectory targeting

`sim.trajectory_targeting` is a headless, reusable first slice of trajectory
targeting. It solves one **instantaneous prograde impulse at periapsis** to reach
first apogee radius, or one **retrograde impulse at apogee** to reach first
perigee radius. Positions are ECI km, velocities km/s, burn magnitude m/s,
and time seconds. The OEL
two-body RK4 propagator evaluates each trial and a finer-step repropagation
checks the accepted result. This is not a general mission-sequence solver.

Run either maintained demonstration with a new, empty evidence directory:

```bash
.venv/bin/python -m sim.trajectory_targeting \
  --spec configs/trajectory_target_apogee_demo.yaml \
  --output-dir outputs/trajectory_target_apogee_full_example \
  --run-maneuver

.venv/bin/python -m sim.trajectory_targeting \
  --spec configs/trajectory_target_perigee_demo.yaml \
  --output-dir outputs/trajectory_target_perigee_full_example \
  --run-maneuver
```

The command saves `targeting_result.json`, `iterations.csv`, and a validated
`post_burn_coast.yaml` scenario. With `--run-coast`, the normal OEL runner
writes `coast_run/review/run.sqlite`; the target packet
records the final-state review query, review-store digest, and target residual.
The generated scenario starts **after** the planned impulse and represents only
the coast; its thrust history therefore shows zero burn. The planned impulse
vector and pre/post-burn states are in the target packet.

With `--run-maneuver`, the tool also generates and validates
`full_maneuver.yaml`. Its initial state is the **pre-burn** state. The native
`fsw.orbit_reference` scheduled-burn path commands the planned ECI delta-v
over one second with ideal wrench hardware, then OEL coasts to the first apsis.
The packet queries `thrust` and `object_state` from the **same** review store,
checks delivered delta-v, locates the first radial-velocity sign crossing,
and interpolates three saved radius samples to estimate the event radius.
It records the bracket and method, so the interpolated event is not presented
as an exact saved state. This is an idealized high-acceleration finite-burn
continuity check, not a finite-burn target solve or propulsion feasibility
claim. A longer burn can miss the ideal impulse target.

For a true instantaneous velocity jump in the OEL run, use `--run-impulsive`.
The generated `exact_impulse.yaml` is validated first. The targeter then
advances a passive ONP run to the exact planned event epoch and applies the
impulse at a sample boundary. `review/run.sqlite` contains the
`impulsive_maneuvers` table with the burn vector and pre/post ECI states at
the same epoch. Its `object_state` row at that sample is post-burn. The
one-burn packet queries the impulse and named apsis from that same store.
There is no modeled duration, thrust force, mass loss, or propulsion system
for this operation.

## Bounded two-burn circular transfer

The experimental `two_burn.v1` workflow varies departure and circularization
prograde impulse magnitudes. It propagates the first burn to first apogee and
achieves both target radius and post-second-burn circular speed. The generic
`sim.trajectory_targeting.vector` corrector accepts named variables, bounds,
scales, named goals and tolerances, and returns Jacobian rank/condition,
iteration history, and a named failure reason. A failed event observation or
singular Jacobian cannot be promoted to a solution.
`sim.trajectory_targeting.trajectory.observe_impulsive_trajectory` is the
matching OEL propagation adapter: up to four exact ECI or RIC impulses,
variable fixed burn epochs and components, and fixed-time or first-apsis
observations of ECI state, radius, speed, and radial speed. Callers pair its
metric tuple with the vector corrector's explicit bounds and tolerances.
The adapter rejects unknown vary fields, time-order conflicts, undefined RIC
geometry, absent events, and resource-limit violations. It remains a
bounded research interface. Its default is point-mass propagation; pass a
trusted scenario mapping as `onp_template` to
`observe_impulsive_trajectory(plan, onp_template=template, vary=changes)`
for the configured ONP forces. Each call uses a fresh OEL runtime, so callers
can pair its metrics with `solve_vector_target` for independent vary controls.
The two-burn CLI below is the first complete validated vector-target workflow.

```bash
MPLBACKEND=Agg .venv/bin/python -m sim.trajectory_targeting.two_burn_cli \
  --spec configs/trajectory_target_two_burn_demo.yaml \
  --output-dir outputs/trajectory_target_two_burn_example \
  --run-impulsive
```

The command writes `two_burn_result.json`, a validated
`impulsive_transfer.yaml`, and a full OEL `run/review/run.sqlite`. The review
verification checks both zero-duration velocity jumps, the first apogee,
and the near-circular state after 600 seconds of further coast. It is limited
to an Earth-centered, coplanar, circular-to-higher-circular, two-body case;
both burns are prograde and bounded. It does not solve arbitrary RIC/ECI
components, departure epoch, plane change, low thrust, or global mission
optimization.

The same packet samples seven bounded flight-time candidates for a phased
circular target, using OEL's existing zero-revolution Lambert solver and
repropagating each Lambert candidate with OEL's two-body integrator. The exact
antipodal Hohmann point is singular in this Lambert formulation, so the
packet uses independent vis-viva at that point. The selected cost is the
minimum of **verified sampled candidates only**. Candidate residuals and
rejection reasons remain visible; this is not a continuous optimality proof.

### Configured ONP force models

The two-burn CLI also accepts a **trusted OEL scenario template**. Its single
`vehicle` must be a passive `trajectory_only` satellite using Earth ECI ONP
with the `two_body` base model. The targeter retains the template's numerical
integrator, orbit substep, epoch, frames, environment, vehicle specs, built-in
perturbations, installed Pro forces, and `objects.vehicle.force_models`
acceleration plugins. It replaces the initial Cartesian state with the
specified circular departure state and selects the study duration, output
directory, and review output. The CLI runs safe validation on the template
and ordinary validation on the generated scenario. Custom modules must be
trusted before the CLI imports them; their acceleration callbacks must be
deterministic for repeated trial and event-root evaluations.

```bash
MPLBACKEND=Agg .venv/bin/python -m sim.trajectory_targeting.two_burn_cli \
  --spec configs/trajectory_target_two_burn_demo.yaml \
  --onp-template configs/trajectory_target_two_burn_onp_j2_demo.yaml \
  --output-dir outputs/trajectory_target_two_burn_onp_j2_example \
  --run-impulsive
```

Every differential-correction trial creates a fresh OEL runtime and propagates
through the same passive ONP dynamics stepper used by the full run. The target
is the first radial apogee radius and the **instantaneous osculating speed**
after burn two. The review check reads both exact impulses from one store and
checks the apogee event, position continuity, delivered delta-v, radius, and
speed. Under perturbations, matching circular speed at one instant does not
make the subsequent orbit persistently circular. The packet records the
template hash and generated scenario. The two-body Lambert time trade is
reported as `not_evaluated` because it does not use the selected forces.

This mode and the generic observation adapter support forces that OEL's Earth ECI ONP satellite runtime can
evaluate, including compatible custom acceleration plugins. It does not
support OGP, CR3BP, interacting objects, finite thrust, attitude-coupled
forces, stochastic or history-dependent plugins, or general optimal control.
The scalar one-burn CLI targeters still use the point-mass path. The maintained GMAT two-burn reference
remains a point-mass comparison; the J2/custom-force checks establish
trial-versus-review consistency within OEL, not independent force-model
accuracy.

`sim.trajectory_targeting.scalar` provides a reusable one-variable,
one-event, one-metric **vary → propagate to event → achieve** interface.
The apsis workflows provide OEL propagator callbacks. The solver uses bounded
differential correction with a bracket safeguard, including decreasing goals.
It reports unreachable delta-v bounds and non-convergence without presenting
those outcomes as feasible plans. Input must be circular or at the
corresponding starting apsis. It rejects the wrong goal direction, absent
first events, and malformed vectors. This scalar path is limited to Earth
point-mass dynamics and the two named one-impulse families. It does not solve
finite burns, J2, drag, resource feasibility, keep-out zones, RPO, general
optimization, or flight decisions.

## GMAT reference

The comparison generator creates a GMAT Earth-point-mass target sequence with
`Vary` → `Maneuver` → `Propagate` to the named apsis → `Achieve`. It compares
the converged GMAT burn, event time, and radius against the exact OEL input
digest. The script,
console, report, executable hash, model assumptions, and fixed tolerances are
recorded in the comparison packet.

```bash
.venv/bin/python -m validation.gmat_apogee_reference \
  --spec configs/trajectory_target_perigee_demo.yaml \
  --oel-result outputs/trajectory_target_perigee_full_example/targeting_result.json \
  --output-dir outputs/trajectory_target_perigee_gmat \
  --generate-only
```

When an installed `GmatConsole` is available, replace `--generate-only` with
`--gmat-bin /absolute/path/to/GmatConsole --gmat-version <release>`. The script
sets GMAT `Earth.Mu` to the OEL value. Inspect the generated script, installed
GMAT version, and frame/event assumptions. A generated script is **pending
reference evidence** until GMAT executes it successfully. The maintained
R2026a results, including source hashes and converged solver consoles, are in
`validation/data/gmat_apogee_target/` and
`validation/data/gmat_perigee_target/`. GMAT agreement is a comparison for
these two cases, not independent physical truth or general targeter
qualification. The analytical vis-viva/event-time tests and the OEL
review-store maneuver checks provide separate evidence.

`validation.gmat_two_burn_reference` generates an independent GMAT sequence
with two `Vary` controls, two impulsive `Maneuver` commands, propagation to
first apogee, and two `Achieve` goals. It compares both solved burn magnitudes,
apogee time/radius, and post-burn circular speed. A generated script alone is
pending evidence until `GmatConsole` executes successfully.

This initial surface is private/experimental. The draft plan in
`Future_Capabilities.md` proposes a public mission-sequence primitive and Pro
optimization later, subject to validation and export review.
