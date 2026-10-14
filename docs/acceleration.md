# Numeric Backends

Version 0.32.0 makes Rust the primary numeric backend for the implemented
propagation, estimator, controller, geometry and standalone analysis selectors.
Python still owns configuration, orchestration, policy and evidence, and remains
available explicitly through `numeric_backend: python`. Trainer uses
`--backend python` for its complete Python path. A missing native wheel or an
unsupported explicitly selected Rust path is an error, not a model substitution.

The core profile requires `oel-rust-orbit` 0.17.x and `oel-rust-game` 0.5.x. Install the qualified wheel for the host before installing
from source, or use the matching signed offline release bundle. Native ABI tags
alone do not establish runtime or physics qualification on another platform.

The separately controlled Numba acceleration layer remains optional. Explicit
Python selection can use that layer where supported.

# Numba Acceleration

Orbital Engineering Lab includes an optional acceleration layer for hot numeric kernels. The first supported backend is
Numba/JIT, exposed as an opt-in feature so ordinary installs and validation runs remain reproducible on machines without
Numba.

## Install

```bash
.venv/bin/python -m pip install -e ".[accel]"
```

For compatibility installs, `.[full]` also includes Numba.
The base package already depends on SciPy, so `.[accel]` adds Numba without
duplicating that core dependency.

## Warmup

Warmup compiles the supported kernels and populates Numba's on-disk cache for the current Python, platform, and CPU.

```bash
.venv/bin/python -m sim.acceleration.warmup
.venv/bin/python -m sim.acceleration.warmup --profile validation
```

The command also works without Numba installed; in that case it exercises the Python fallback kernels and reports the
backend as `python`.

## Benchmark

Use the bundled benchmark to confirm the supported RK4/J2 orbit fast path on a local machine:

```bash
.venv/bin/python -m sim.acceleration.benchmarks --iterations 10000
.venv/bin/python -m sim.acceleration.benchmarks --iterations 10000 --json
```

The benchmark reports baseline Python propagator time, accelerated-path time, speedup, and final state delta norm.
Use `--kind attitude` to include the attitude exponential-map propagation benchmark, `--kind estimation` to time the
orbit/attitude EKF finite-difference Jacobian paths, or `--kind all` to run every local acceleration benchmark:

```bash
.venv/bin/python -m sim.acceleration.benchmarks --kind all
.venv/bin/python -m sim.acceleration.benchmarks --kind estimation --estimation-iterations 1000
```

For end-to-end timing across propagation, the full satellite loop, sensing and estimation, actuators, lifecycle models,
campaign orchestration, and artifact generation, use the [full-path performance suite](performance-benchmarks.md).

## Runtime Controls

Acceleration is disabled by default. Enable it in YAML:

```yaml
simulator:
  acceleration:
    mode: auto   # off | auto | numba
    warmup: false
```

The environment variable `OEL_ACCELERATION=off|auto|numba` overrides config mode for the current process.
Game sessions force acceleration off through their runtime config so players do not pay first-run JIT cost.

## Current Coverage

The fixed-step RK4 orbit fast path covers propagation when the dynamics use only:

- two-body gravity
- J2
- J3
- J4
- constant command acceleration for the integration step

Normalized spherical-harmonic gravity fields also use an accelerated force-evaluation kernel with fixed-step and
adaptive integrators. The authoritative Python frame implementation still prepares each ECI-to-body-fixed rotation;
the optional backend compiles the normalized Legendre recurrence and degree/order summation. Unnormalized or mixed
coefficient sets retain the existing Python finite-difference implementation. Other force plugins continue through
their existing implementation even when they coexist with an accelerated normalized gravity field.
Repeated IAU-76/80/EOP frame inputs reuse an exact immutable cached rotation;
callers still receive independent arrays, and no time interpolation or frame
approximation is introduced.

NRLMSISE-00 uses an accelerated upper-atmosphere diffusion/spline kernel across
its full altitude and space-weather input domain. The kernel preserves the
reference coefficient set and operation ordering, runs with fast math disabled,
and falls back to the authoritative Python implementation whenever acceleration
is unavailable or disabled. A larger no-fast-math kernel covers the standard-
switch thermosphere at and above 300 km when the model's historical Ap inputs
select its quiet branch; disturbed conditions and lower altitudes retain the
complete authoritative path. WGS-84 atmosphere coordinates use the same exact
iterative conversion in one compiled boundary for every accelerated atmosphere
model. MSIS-86 similarly accelerates its complete
temperature/density profile calculation and adds a compiled fixed-switch globe
path when the model's Ap-history criterion selects its quiet branch. Disturbed-
Ap inputs retain the authoritative Python globe calculation, while all MSIS-86
paths retain the Python fallback when acceleration is unavailable or disabled.
Other atmosphere models retain their existing model-specific accelerated
kernels and Python fallbacks.

DE440 light ephemerides accelerate the mandatory Earth-Moon barycenter, Moon,
and Sun Chebyshev evaluations in one no-fast-math kernel when the compact NPZ
coefficient format is selected. The Sun/Moon-only path also performs UTC-to-TDB
conversion and geocentric reduction inside that boundary. Callers that need
only the geocentric Sun/Moon pair avoid constructing the complete position dictionary; optional planetary
bodies and MAT-file coefficient sources retain the authoritative Python paths.
Cannonball SRP similarly fuses spacecraft-to-Sun geometry, cylindrical or
conical eclipse evaluation, pressure scaling, and acceleration into one
no-fast-math kernel whenever the time-dependent environment has resolved a Sun
position. Acceleration-off mode retains the original Python implementations for
both capabilities.

Fixed-step RK4 scenarios in the exact quiet-thermosphere NRLMSISE-00 domain can
also execute the complete supported force plan behind one compiled boundary.
The plan preserves configured plugin order and may combine normalized spherical
harmonics, drag, cannonball SRP, and DE440 Sun/Moon third-body gravity beneath
the existing `OrbitPropagator.propagate` API. Frame/EOP and ephemeris
preparation remain owned by their authoritative implementations. Disturbed Ap,
altitudes below 300 km, and acceleration-off mode retain the authoritative
implementations.

A staged compiled-component tier covers richer plans outside that fused domain
for fixed-step RK4 and adaptive RKF78 (`rkf78`/`adaptive`) or Dormand-Prince
(`dopri5`) propagation. It supports explicit J2/J3/J4,
normalized spherical harmonics, drag and lift with constant density or any OEL
atmosphere family (exponential, USSA-1976, NRLMSISE-00, MSIS-86, Jacchia-70,
JB2006, JB2008, and Harris-Priester), cannonball SRP, Sun/Moon and selected
planetary third-body gravity, and plans interleaved with custom acceleration
callbacks. Authoritative Python owners prepare state-dependent density, frame,
ephemeris, and custom-plugin values; compiled kernels evaluate the supported
numeric force components; the propagator then accumulates all contributions in
the configured plugin order. Custom Python callbacks are preserved rather than
silently treated as nopython code. Small force plans continue through their
faster specialized compiled evaluators when staging would add overhead, and
mixed or unnormalized harmonic fields retain their authoritative evaluator.

RIC frame transforms are wired into the runtime acceleration path, and re-entry scalar kernels are available for
warmup and parity tests while they are staged for broader integration.

Attitude acceleration covers the exponential-map rigid-body path used by
`OrbitalAttitudeDynamics`. A staged numeric plan can evaluate the built-in
gravity-gradient, magnetic, scalar/facet drag, and scalar/facet SRP torques and
propagate all attitude substeps behind one compiled boundary. The plan preserves
the public disturbance accumulation order, quaternion normalization,
angular-rate/torque clamps, singular-inertia handling, and guardrail event
accounting. Custom disturbance objects, geometry lookup profiles, rectangular-
prism face models, acceleration-off runs, and unavailable acceleration backends
retain the authoritative Python path.
The coupled dynamics object's owned default two-body orbit propagator inherits
the same acceleration mode; explicitly supplied orbit propagators retain their
own configured mode.

Estimator acceleration currently covers the orbit EKF two-body RK4 propagation/Jacobian path and the attitude EKF
propagation/Jacobian path used inside the joint-state estimator. This targets long RIC_PD-style runs where estimator
updates dominate runtime after the core orbit and attitude dynamics are accelerated.

## Passive Rust Sampled Histories

For passive ECI two-body scenarios using only built-in J2/J3/J4 terms, the
native sampled-history path also covers `rkf78`, its `adaptive` alias, and
`dopri5`. It carries adaptive step suggestions and diagnostics across the
ordinary scenario sample boundaries. Wheel 0.13.0 also batches adaptive passive
full-force contexts with normalized harmonics, drag, SRP, and Sun/Moon forces;
frame and atmosphere callbacks retain their existing owners. Numeric tables,
environment inputs, and resource metadata bind each prepared history.
Attitude, resources, custom forces,
system forces, and other active dynamics continue through their normal
per-step paths. Changing cadence or state invalidates the prepared history.

## Rust CR3BP

The separately installed `oel_rust_orbit` wheel (0.6.0 or newer) also supports
OEL's ideal CR3BP in physical rotating barycentric coordinates:

```yaml
simulator:
  dynamics:
    orbit:
      model: cr3bp
      cr3bp_system: earth_moon
      numeric_backend: rust
      integrator: rkf78  # rk4 and dopri5 are also supported
```

Rust is the default. Native state and reference/STM integration use
the existing equations, physical units, tableaus, adaptive step control and
diagnostic shape. Direct `propagate_cr3bp_state` and
`propagate_cr3bp_reference_stm` accept `numeric_backend="rust"`; the latter
integrates all 42 components on a common step sequence. Scenario selection
does not change halo-seed phase initialization or Trainer preview defaults.
Validation rejects a missing or older wheel. See [CR3BP research](cr3bp-research.md)
for frame and model limits. Measured kernel timing does not establish a
whole-simulation or graphics speedup.

## Resource Notes

Acceleration reduces runtime per supported numerical step; it does not replace
resource planning, checkpointing, or thermal safeguards for long campaign
workflows. The first accelerated call may include compilation overhead unless
kernels have already been warmed.

## Rust Estimation Flow Batches

Wheel 0.13.0 batches nonlinear relative ONP finite-difference deputy flows and
orbit UKF sigma trajectories when both the estimator/workflow and propagator
select Rust, with ECI RK4 and only built-in ordered J2/J3/J4 forces. Python
retains frame transforms, filter updates, residuals, and evidence construction.
Unsupported force/plugin owners and older wheels retain scalar propagation.

## Rust P.676 Arithmetic

The optional Pro communications path now passes `numeric_backend="rust"`
through P.676-13 specific attenuation and Annex 2 slant attenuation. Python
loads and content-binds the edition coefficient tables, validates applicability
and broadcasting, interpolates effective-height coefficients, and constructs
RF ledgers and closure decisions. This requires wheel 0.13.0. The default is
Rust; missing native symbols fail native selection. Floating-point
spectral reductions are checked to a bounded tolerance, not bitwise equality.
This numerical port adds no new atmospheric applicability or qualification.


## Prepared Rust Workflows (0.14.0)

`SequentialODConfig(numeric_backend="rust")` batches EKF nominal plus six
forward-difference trials and UKF sigma trajectories using the same ECI RK4
and ordered built-in J2/J3/J4 envelope. A supplied propagator must also select
Rust; the default propagator inherits this explicit workflow selection. Filter
updates, process noise, maneuver gates, RTS smoothing and checkpoints stay in
Python. Default Checkpoint identities retain their numeric backend; Rust selection
is bound into its checkpoint configuration. Unsupported force owners and older
wheels retain ordinary scalar propagation.

Rich coverage's existing Rust selector batches boundary-point WGS84 geodesy.
Ray intersections retain reference arithmetic because boundary coordinates enter
the product hash byte-for-byte. No coverage product contract changes.

The optional Pro `schedule_downlink_opportunities(..., numeric_backend="rust")`
forwards to the existing exact bounded native tasking search. Python retains
source-link bindings, value-ranked candidate preselection, resource validation,
selected data totals and schedule hashes. This remains a Pro workflow and adds
no globally optimal claim beyond the selected candidate bound.

Passive ONP OD force-plan contexts also batch RKF78/adaptive and DOPRI5 arcs,
and support `state`, `drag_scale`, `cd_scale` and `srp_scale` fits. Python's
single parameter-mapping owner prepares every candidate's object specs; each
distinct geometry goes through normal scenario validation. A bounded cache
reuses up to eight parameter contexts across state trials. Every adaptive
candidate starts afresh and carries step suggestions only within that arc.
Native admission compares the configured nominal arc against SimulationSession;
unsupported plans, attitude/resource coupling, maneuvers and invalid/impact
candidates retain ordinary session evaluation. Python retains optimization,
loss, priors, clipping, rank, holdouts and artifacts. Joint state/scale fits
extend the established fixed native Jacobian policy with 1e-4 scale steps;
parameter-only fits retain their existing derivative policy. Report the selected
evaluator and convergence/holdout gates separately from timing.

`RCSAllocationAwareController(numeric_backend="rust")` prepares content-bound
thruster matrices and fuses achieved force, torque and inertial-force products
through `RCSGeometryContext`. Mutable directions, lever arms, bounds or names
invalidate the cache. NumPy least squares and one-bound-at-a-time freezing remain
authoritative. Older wheels use the previous three native matrix-vector calls.
Rust is the default for these numerical workflows. Local performance evidence
is workload-specific and does not establish external physics qualification.

## Integrated Numerical Workflows (0.15.0)

The native wheel provides the primary numerical backend and is a public-core
dependency. Python continues to own workflow policy; explicit Python numerical
selection remains available for reference comparisons.

`run_integrated_relative_od(..., numeric_backend="rust")` forwards the selector
to batch HCW/SS-J2/TH/YA fits, sequential filters, nonlinear holdout predictions,
multi-arc fits and the relative ONP variational baseline. Its default chief
propagator inherits the selector. A supplied propagator keeps its own backend
and force-model ownership; eligible Rust RK4 chief covariance trials are batched.
Reports distinguish the workflow selection from the supplied chief backend.
Direct-from-epoch transitions, solver policy, model ranking and evidence remain
unchanged. Native fits have bounded numerical parity; optimizer termination text
and floating-point diagnostics can differ. This remains a Pro OD workflow.

Native P.676 constant-profile evaluation prepares one spectral result for equal
frequency, pressure, temperature and density across samples. Reuse binds all four
inputs and the native coefficient-table context, with at most eight cached
profiles. Annex 2 heights are prepared once per batch when interpolated
coefficients are also equal; only elevation arithmetic repeats. Variable profiles
and older compatible P.676 wheels retain ordinary native batches. Applicability,
coefficient interpolation and atmospheric ledgers remain Python responsibilities.

`refine_time_of_closest_approach(..., numeric_backend="rust")` uses immutable
operation-local history contexts and batches interval interpolation and Hermite
candidate arithmetic. `assess_histories` and `assess_conjunction` forward this
geometry selector through baseline, candidate and secondary assessments. It
selects geometry only; propagation and targeting retain their existing owners.
The explicit Python TCA also prepares history arrays once per operation, retaining
the public `StateHistory.arrays()` independent-copy behavior. NumPy polynomial
roots, impulse sides, candidate ordering, ties, encounter decisions and adaptive
Pc quadrature remain Python-owned. Native histories and each crossing are bounded
to one million samples/cases. Missing new geometry symbols fail explicit selection.

`AttitudeEKFEstimator(..., numeric_backend="rust")` uses `AttitudeEKFContext` to
evaluate nominal plus seven forward-difference Euler predictions in one crossing.
This preserves the estimator's normalized Euler quaternion step, rather than
substituting the rigid-body exponential-map step. Inertia LU preparation is
content-bound with eight-entry retention; singular solves preserve
`numpy.linalg.LinAlgError`. Measurements, epoch splitting, antipodal alignment,
Joseph covariance update and innovation-solve fallback remain Python-owned.
`JointStateEstimator(..., attitude_numeric_backend="rust")` selects only that
attitude owner. Missing symbols fail explicit selection. Compare native timing
with warmed Numba when choosing a backend; no general speed ranking is implied.


## Design Workflows And Probability (0.16.0)

Constellation design accepts optional `propagation.numeric_backend="rust"` in
its problem, forwarding to existing ONP, coverage and directed-link kernels.
Pro searches inherit it through `public_evaluation.propagation`, including resume
and replay. Pro trajectory optimization accepts the same propagation setting
and batches uninterrupted fixed-duration RK4 coasts through existing zonal history
arithmetic, up to 4096 steps per crossing. Fractional last widths are retained.
Sample preparation removes unused orbital-element calculations and preserves
elliptical-state guards. Full event/terminal metrics and scalar event, finite-burn
and adaptive owners retain their policy and selected backend. Rust is the
default; both selections bind problem and replay identities.

`collision_probability_2d(..., numeric_backend="rust")` uses a stateless native
integrand capsule with SciPy's low-level callback ABI. Python retains covariance
validation, SciPy adaptive quadrature, fine/coarse convergence evidence and
acceptance. The existing geometry selector stays independent: use
`probability_numeric_backend="rust"` on history/avoidance assessments or
`numeric_backend="rust"` on instantaneous CDM assessment to select native Pc.
Native probability evidence records its backend; diagnostic floats can differ
within tested bounds. Missing required kernels fail explicit selection.

## Analysis workflows (0.17.0)

Collection opportunities, spacecraft power, and orbit
lifetime accept a top-level `numeric_backend="rust"`. Collection
uses native pointing/gimbal and analytic Sun arithmetic; power uses native
analytic Sun arithmetic; lifetime prepares a native RK4 force/stage context
once per propagator for constant/exponential atmospheres; other supported
atmospheres use the native integrator with their existing force callbacks.
Python keeps workflow decisions and events. Rust is the
default and native selection fails closed when required symbols are
unavailable. Problem normalization retains both backend selections.

The entitled Pro tracking workflow accepts `propagation.numeric_backend="rust"`
for both fit and holdout propagation while preserving Python measurement and
filter policy. See each workflow's documentation for its admitted envelope.
