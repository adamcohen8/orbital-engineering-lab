# Event-Driven Trajectory Targeting

OEL v0.29 adds a bounded public trajectory-design primitive for deterministic
coast/burn sequences. It can propagate to a fixed duration or a state event,
apply ECI or RIC impulsive burns, correct declared burn or coast-time variables
against terminal constraints, and independently repropagate every accepted
solution through the OEL Numerical Propagator (ONP).

This is a transparent single-shooting tool. It is not a general optimizer or
an operational maneuver-planning system.

## Quick Start

Solve the bundled Hohmann-style apoapsis target and retain its JSON evidence:

```bash
python -m sim.trajectory_design solve \
  examples/trajectory_targeting/hohmann_apoapsis.json \
  --output outputs/trajectory_targeting/hohmann_evidence.json
```

Execute only the supplied initial decision vector, without differential
correction:

```bash
python -m sim.trajectory_design propagate \
  examples/trajectory_targeting/hohmann_apoapsis.json
```

The Python entry points are available from `sim.trajectory_design`:

```python
import json
from pathlib import Path

from sim.trajectory_design import TrajectoryTargetingProblem, solve_trajectory_target

raw = json.loads(
    Path("examples/trajectory_targeting/hohmann_apoapsis.json").read_text()
)
problem = TrajectoryTargetingProblem.from_mapping(raw)
evidence = solve_trajectory_target(problem)
assert evidence["authoritative_repropagation"]["status"] == "verified"
```

Python callers can select `solve_trajectory_target(problem, numeric_backend="rust")`
with wheel 0.13.0 or newer for RK4 finite-difference trials using two-body
and ordered built-in J2/J3/J4 forces. Fixed-duration trials use one native
batch. Event coasts prepare bounded native coarse samples while Python retains
crossing detection, angular unwrapping, refinement decisions, failure receipts,
and propagation accounting. Direct `execute_trajectory` callers may select
`numeric_backend="rust"` for the same fixed-step RK4 envelope. Direct Rust
trajectory execution and Rust targeting Jacobians reject `rkf78`, `dopri5`,
and the `adaptive` alias before native probing or trajectory work; they do not
fall back automatically. Select `numeric_backend="python"` to execute those
explicit adaptive/non-RK4 paths. The solver retains Python iteration, line
search, terminal constraints, and independent authoritative repropagation. The
default is `numeric_backend="rust"`; select `numeric_backend="python"` for the
independent Python reference implementation.

## Problem Contract

The input contract is `oel.trajectory_targeting_problem.v1`. State values are
Earth-centered ECI Cartesian `[x, y, z, vx, vy, vz]` in km and km/s. Burn
commands and burn decision variables are in m/s.

Each problem contains:

- one initial Cartesian state;
- one or more ordered `coast` or `impulsive_burn` segments;
- zero or more decision variables;
- one or more terminal equality constraints;
- explicit propagation, event-location, and solver settings.

The parser rejects unknown segment types, duplicated names or variable targets,
nonpositive tolerances, ambiguous coast termination, invalid burn fields, and
underdetermined problems with more decision variables than constraints.

## Mission Segments

An impulsive burn declares `frame: eci` or `frame: ric` and a three-component
`delta_v_m_s`. RIC components are ordered radial, in-track, cross-track and are
resolved from the state immediately before the burn.

A coast declares exactly one stopping form:

- `duration_s` for fixed-duration propagation; or
- `stop` for state-event propagation.

An event stop includes a quantity, target, crossing direction, maximum search
duration, and optional minimum elapsed time. The minimum elapsed time is useful
for an apsis event whose radial velocity is already zero at the initial state.
The event locator brackets the crossing on ONP propagation steps, then
repropagates and bisects inside the bracket to the declared time/value
tolerances. Event search begins at the exact minimum-elapsed boundary, angular
events follow a continuous angle branch across 0/360 degrees, and an exhausted
refinement budget fails closed rather than returning an under-refined event.

Supported event and terminal quantities are:

- ECI position and velocity components;
- radius, altitude, speed, and radial velocity;
- semi-major axis and eccentricity;
- inclination, RAAN, argument of periapsis, and true anomaly;
- elapsed mission time.

Angular residuals use the shortest signed difference across the 0/360-degree
boundary.

## Decision Variables And Constraints

Burn variables name one component on one burn segment:

- ECI: `delta_v_x_m_s`, `delta_v_y_m_s`, `delta_v_z_m_s`;
- RIC: `delta_v_r_m_s`, `delta_v_i_m_s`, `delta_v_c_m_s`.

A fixed-duration coast may expose `duration_s` as a timing variable. Every
variable declares a positive finite-difference perturbation in that variable's
native units.

Each terminal constraint declares a target and a positive tolerance. The
corrector works on normalized residuals:

`normalized residual = (actual - target) / tolerance`

A constraint is satisfied when the absolute normalized residual is at most
one. The evidence retains the actual value, dimensional residual, normalized
residual, and disposition for every constraint.

## Corrector And Repropagation

The public corrector uses central finite differences and a least-squares Newton
step. Every iteration records:

- the complete decision vector;
- raw and normalized residual vectors;
- the finite-difference Jacobian;
- the effective finite-difference perturbation used for each variable, including
  a reduced central perturbation when a coast duration is near its positive domain boundary;
- Jacobian rank and singular values;
- the proposed correction;
- the accepted line-search scale and resulting residual norm.

The solver fails closed on a missed event, exhausted event refinement, a failed
or rank-deficient Jacobian, an iteration limit, or the absence of an improving
line-search step. It does not silently reinterpret these outcomes as partial
success.

After convergence, OEL creates a fresh propagator and executes the materialized
sequence again from the original state. A result is `converged` only if this
authoritative repropagation also satisfies every terminal constraint. The
evidence records final-state differences between the shooting pass and the
fresh repropagation.

## Evidence And Resource Ledger

The output contract is `oel.trajectory_targeting_evidence.v1`. A successful
packet includes:

- a SHA-256 identity for the normalized problem;
- convergence history and terminal-constraint rows;
- the materialized segment timeline and event receipts;
- burn commands in their declared frame and resolved ECI frame;
- burn count, total delta-v, coast time, ONP step count, trajectory evaluations,
  and Jacobian evaluations, including refinement work and failed event attempts;
- the independent authoritative repropagation and its constraint evaluation;
- explicit limitations and non-claims.

Failure packets preserve the best available trajectory/constraint evidence and
use explicit statuses such as `missed_event`, `rank_deficient`, `infeasible`,
or `non_convergent`.

## Validation Envelope

The checked-in acceptance suite covers:

| Claim slice | Validation anchor |
| --- | --- |
| Event location | Two-body apoapsis radial-velocity crossing with a refined bracket |
| Impulsive targeting | Closed-form Hohmann departure and plane-change burns |
| Orbital-element/timing targets | Closed-form phasing-energy target and exact coast-duration target |
| Cartesian rendezvous target | Known two-axis burn recovered from an independent terminal state |
| Jacobian | Direct component/unit check plus pinned Orekit central-difference comparison |
| Propagation | Energy/angular-momentum conservation, one-period reversibility, pinned Orekit 13.1.7 states |
| Failure handling | Fixed infeasible, underdetermined, rank-deficient, missed-event, and iteration-limited fixtures |
| Solution verification | Fresh ONP repropagation required for every accepted solution |

The Orekit fixture is bound to the public Java generator source, retained raw
generator CSV, and pinned runtime hashes. It requires no STK, ODTK, or paid service. Independent
implementations are compared within reviewed residual envelopes rather than
being forced to machine-zero parity.

Run the focused acceptance suite with:

```bash
python -m pytest \
  sim/tests/test_trajectory_targeting.py \
  sim/tests/test_trajectory_targeting_open_reference.py
```

## Public And Pro Boundary

Public OEL includes this deterministic executor, event definitions, one
transparent single-shooting corrector, small examples, failure evidence, and
authoritative solution repropagation.

Pro remains the home for bounds and inequality/path constraints, finite-burn
optimization, multiple shooting or collocation, multi-start/global search,
campaign-scale robustness studies, stationkeeping or launch-window workflow
packs, design dashboards, and customer-specific vehicle/constraint libraries.

No public result claims global optimality, uncertainty robustness, collision
safety, operational readiness, maneuver authorization, or flight
qualification.
