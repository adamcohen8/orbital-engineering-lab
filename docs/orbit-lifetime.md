# Deterministic Orbit Lifetime Analysis

OEL v1.0 can propagate one low-orbit spacecraft with ONP drag, retain the
declared atmosphere and spacecraft assumptions, refine warning, disposal, and
reentry-altitude crossings, and write replayable JSON/CSV evidence. It can also
compare atmosphere models while holding every non-atmosphere input fixed.

## Run the canonical workflow

```bash
.venv/bin/python examples/python/orbit_lifetime_workflow.py \
  --output-root outputs/public_orbit_lifetime
```

The example runs a deliberately accelerated synthetic decay case, compares
four public atmosphere models, authoritatively replays both products, and
retains the single-case summary in a verified study-lifecycle bundle. The
terminal record separately reports scientific replay and lifecycle identity
replay.

## Analyze one declared case

Copy the checked-in problem and deliberately set the epoch, ECI state,
spacecraft, atmosphere inputs, horizon, cadence, and altitude thresholds:

```bash
.venv/bin/python -m sim.orbit_lifetime validate \
  my_orbit_lifetime_problem.json
.venv/bin/python -m sim.orbit_lifetime analyze \
  my_orbit_lifetime_problem.json \
  --output-dir outputs/my_orbit_lifetime
.venv/bin/python -m sim.orbit_lifetime replay \
  outputs/my_orbit_lifetime
```

The same commands are available from the unified launcher as
`oel lifetime ...`. The destination must not already exist. Inputs and outputs
use the versioned [Orbit Lifetime Analysis Contract](contracts/orbit-lifetime-contract.md).

## Compare atmosphere assumptions

```bash
.venv/bin/python -m sim.orbit_lifetime validate-comparison \
  my_atmosphere_comparison.json
.venv/bin/python -m sim.orbit_lifetime compare \
  my_atmosphere_comparison.json \
  --output-dir outputs/my_atmosphere_comparison
.venv/bin/python -m sim.orbit_lifetime replay-comparison \
  outputs/my_atmosphere_comparison
```

Each case changes only the normalized atmosphere record. OEL does not fetch
weather: constant/exponential parameters, F10.7, and Ap values are frozen in
the problem and bound by the evidence digest.

## Read the result

Start with `orbit_lifetime_summary.json`:

- `outcome` distinguishes `reentry_threshold_reached` from
  `horizon_complete`, `atmosphere_domain_limit_reached`, or
  `earth_surface_reached`;
- `thresholds` records whether and when each declared altitude was crossed;
- `initial`, `final`, and `changes` show the bounded osculating-orbit change;
- `extrema` summarizes retained output samples;
- `energy_accounting` exposes drag-work closure when J2 is off;
- `atmosphere_effective` records the effective model domain and discrete
  Harris-Priester table when applicable;
- `implementation_identity` binds the analysis source and atmosphere assets;
- `resource_use` records steps, samples, events, and propagated time; and
- `claim_limits` travels with the result.

Use the timeseries for altitude, density, drag acceleration, semi-major axis,
perigee, and apogee histories. Use the event CSV for the refined thresholds and
their integration brackets. Cite a result only after replay returns
`status: verified`.

`horizon_complete` means only that no configured stopping threshold ended the
declared horizon. It is not a predicted infinite lifetime, and the analyzer
does not extrapolate beyond its final authoritative state. Refined thresholds
are instantaneous geocentric-altitude crossings, not regulatory findings or
surviving-debris assessments.

For `stop_at_reentry: false`, propagation never continues outside the selected
atmosphere model or Earth-surface domain. It terminates with the corresponding
explicit domain outcome.

## Local Edition And Private-Service Boundary

The strict contracts, deterministic local ONP single cases, frozen atmosphere
inputs, identical-input comparisons, JSON/CSV evidence, authoritative replay,
schema, validation fixtures, and small examples are public. The bounded local
workflow does not include managed current/historical weather ingestion, density
and ballistic-coefficient calibration, customer data governance, compliance
packages, or operational qualification evidence. Hosted service infrastructure
and the Manifold remain private. This v1 result is not operational authority.

## Optional Rust propagation

Top-level `numeric_backend: "rust"` selects a prepared native ONP RK4 force/stage
context, reused within each lifetime propagator. Wheel 0.17.0 or newer is
recommended. The admitted scope is constant or exponential atmosphere, drag
on/off, and optional J2. Other lifetime atmospheres retain their OEL stage callbacks through the native
integrator when they cannot prepare a fused force context.
Python still decides every threshold, interior radial minimum, refinement,
domain limit, termination, sample, and drag-work interval; no steps are batched
past a decision. Rust is the default; explicit Python selects the reference implementation.
Both selections persist into normalized problems, comparisons, and replay;
lifetime implementation identity includes the native source and adapter files.
Numerical parity and fixed-state external comparisons are bounded evidence,
not independent propagated-lifetime qualification.
