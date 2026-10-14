# Orbit Dynamics

This page documents the current public orbit-propagation model in Orbital
Engineering Lab. It is intentionally conservative: it describes the equations,
configuration surface, implementation locations, evidence hooks, and limits
that can be traced to the checked-in public source.

OEL orbit propagation is a deterministic simulation model for review,
education, and engineering prototyping. It is not a flight-qualified orbit
determination or mission-operations authority.

## Model Scope

The normal satellite dynamics path propagates an inertial Cartesian state:

```text
x = [r_eci_km, v_eci_km_s]
```

where position is in kilometers, velocity is in kilometers per second, and time
is in seconds. The default central body is Earth, with constants from
`sim/dynamics/orbit/environment.py`.

The baseline force model is two-body point-mass gravity plus optional command
acceleration:

```text
dr/dt = v
dv/dt = -mu * r / |r|^3 + a_cmd
```

OEL may add optional perturbation accelerations when the scenario enables them:

- J2, J3, and J4 zonal gravity terms,
- spherical-harmonic gravity terms from inline or user-provided coefficients,
- atmospheric drag and coefficient/vector-based lift,
- solar radiation pressure with eclipse/shadow handling,
- Sun and Moon third-body accelerations,
- CR3BP propagation for the explicitly selected `model: "cr3bp"` path.

Product terminology:

- **OGP**, the **OEL General Propagator**, is OEL's catalog-style
  general-perturbations family for TLE/mean-element products. **OGP-SGP4** is
  the near-Earth SGP4 path, and **OGP-SDP4** is the deep-space/resonance path
  for TLEs at or above the 225-minute period threshold.
- **ONP**, the **OEL Numerical Propagator**, is OEL's native numerical
  propagation path with the selected force-model profile.

ONP should not be described as HPOP; HPOP refers to external
reference/validation tooling when those assets are available.

When `model: "cr3bp"` is selected, the scenario validator rejects the
two-body perturbation flags that belong to the Earth-centered special
perturbations path.

CR3BP is the explicit exception to the normal ECI state contract. Its
six-component state and command acceleration are Earth-Moon synodic rotating
frame quantities. The generic runtime `StateTruth` and `Command` attribute
names retain their historical `*_eci_*` spelling for compatibility, but run
evidence labels CR3BP state histories and command acceleration as
`cr3bp_rotating`; consumers must use that frame metadata rather than infer a
frame from the compatibility attribute names.

## Numerical Propagation

The runtime propagator is `sim/dynamics/orbit/propagator.py::OrbitPropagator`.
For the normal two-body/special-perturbations path, it builds the derivative

```text
dr/dt = v
dv/dt = a_two_body + a_command + sum(a_plugins)
```

and advances the state with one of the configured numerical integrators from
`sim/dynamics/orbit/integrators.py`:

- `rk4`: fixed-step fourth-order Runge-Kutta, the default.
- `rkf78` or `adaptive`: adaptive Fehlberg 7(8)-style stepping.
- `dopri5`: adaptive Dormand-Prince 5(4)-style stepping.

These selections also apply to the CR3BP runtime path. Its state and augmented
reference/STM helpers accept the same integrator and tolerance options; see
[CR3BP research](../cr3bp-research.md#adaptive-cr3bp-integration).

The outer simulator step is controlled by `simulator.dt_s`. If
`simulator.dynamics.orbit.orbit_substep_s` is set, `OrbitalAttitudeDynamics`
breaks each outer step into smaller orbit substeps before calling the
propagator. Adaptive integrators may take internal accepted and rejected steps
inside a single outer/substep interval, and the propagator exposes adaptive step
accounting on `last_adaptive_step_info` and `adaptive_step_info`.

`sim/dynamics/orbit/two_body.py` also provides `propagate_two_body_rk4`, a
small RK4 helper used by selected controllers, estimators, and mission-analysis
modules. It uses the same two-body acceleration form plus a constant command
acceleration over the step.

The optional acceleration layer has specialized, fused, and staged force-plan
tiers. Profitable staged plans can cover fixed-step RK4 and adaptive RKF78,
`adaptive`, or `dopri5` integration with J2/J3/J4, drag/lift, normalized
spherical harmonics, SRP, and Sun/Moon/planetary third bodies. Acceleration
availability, force-plan size, coefficient normalization, atmosphere,
EOP/ephemeris inputs, and supported callback boundaries determine routing.
Custom callbacks remain Python-authoritative and may be interleaved with
compiled built-in components; unsupported plans fall back to the Python path.
The public `objects.<id>.force_models` contract exposes these additive ONP
callbacks through trusted scenario pointers. Each receives a read-only ECI
state in km and km/s plus an absolute UTC Julian date, and returns ECI
acceleration in km/s². See [scenario YAML](../scenario-yaml.md#custom-onp-force-models).
For two-body reciprocal forces, the opt-in synchronized system-force path
integrates a trajectory-only spacecraft pair at shared RK4 stage times. The
object-scoped optional context exposes time-labelled committed partner truth;
it is not a substitute for the synchronized path.

## Frames, Units, And Signs

OEL's normal propagated satellite truth state is ECI-like Cartesian state:

- `position_eci_km`: ECI position, kilometers.
- `velocity_eci_km_s`: ECI velocity, kilometers per second.
- `thrust_eci_km_s2`: commanded translational acceleration, kilometers per
  second squared.
- `mu_km3_s2`: gravitational parameter, kilometers cubed per second squared.
- spacecraft mass is in kilograms, area is in square meters, drag/SRP
  acceleration is converted back to kilometers per second squared.

Central gravity points toward the central body through `-mu * r / |r|^3`.
Atmospheric drag opposes atmosphere-relative velocity. SRP acceleration uses
the configured Sun geometry and shadow factor; the implementation applies the
sign convention in `accel_srp` rather than exposing a separate user-facing
force direction switch.

Frame conversion helpers live in `sim/dynamics/orbit/frames.py`. Scenario
YAML can set `simulator.frames.model` to the default `simple_gmst` path or to
`iau76_80_eop` with an EOP file for HPOP/MATLAB-style parity studies. OEL
records the resolved model, EOP path, and time-scale assumptions in
`frame_provenance`; public docs should not claim an EOP-backed high-precision
frame reduction unless that provenance shows the corresponding frame model and
data paths.

TLE initialization is separate from propagation. By default, OEL samples OGP
(`OGP-SGP4` or `OGP-SDP4`, depending on the TLE period) to recover an
ECI-compatible initial state; subsequent propagation uses the configured ONP
force model. Passive catalog-style OGP propagation is available only for
objects that explicitly set `propagation_method: general` and
`general.model: sgp4`, with the limitations documented in
`docs/scenario-yaml.md` and `docs/known-limitations.md`.

## Configuration Surface

The main YAML controls live under `simulator.dynamics.orbit`:

```yaml
simulator:
  dt_s: 1.0
  dynamics:
    orbit:
      model: "two_body"
      orbit_substep_s: null
      integrator: "rk4"
      adaptive_atol: 1.0e-9
      adaptive_rtol: 1.0e-7
      j2: false
      j3: false
      j4: false
      drag: false
      srp: false
      third_body_sun: false
      third_body_moon: false
      spherical_harmonics:
        enabled: false
```

Common related controls include:

- `simulator.acceleration.mode`: `off`, `auto`, or `numba` for optional numeric
  acceleration.
- `simulator.environment.atmosphere_model` (or explicit `density_kg_m3`) and
  `simulator.environment.atmosphere_env`: the primary shared density-model
  selection and inputs for drag/lift. The orbit-level `atmosphere_model` is a
  legacy direct-runtime compatibility alias and does not satisfy validate-first
  scenario ownership by itself.
- object `specs.mass_kg`, `drag_area_m2`, `area_m2`, `cd`, `srp_area_m2`,
  `cr`, and related aero/SRP fields: spacecraft properties used by drag, lift,
  and SRP.
- `simulator.initial_jd_utc`: run epoch used by TLE initialization and by
  epoch-aware environment/frame paths when configured.
- `spherical_harmonics.terms`, `coeff_path`, `degree`, `order`,
  `reference_radius_km`, `frame_model`, and `eop_path`: spherical-harmonic
  gravity inputs.

Mode-driven analytic, DE440, and SPICE Sun/Moon ephemerides require an
absolute epoch. Without one, the mode branch is not activated and the resolver
uses fixed default vectors (Sun at `[AU_KM, 0, 0]`, Moon at
`[384400, 0, 0]` km) unless explicit vectors, sampled histories, or a callback
are supplied.

Scenario validation enforces strict boolean values for orbit booleans and
rejects unsupported CR3BP/perturbation combinations. Users should validate new
or edited scenarios before running:

```bash
.venv/bin/python run_simulation.py --config <path> --validate-only
```

For reviewable orbit evidence, enable the review store:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

Then inspect `object_state`, `relative_state`, `thrust`, `events`, `metrics`,
and `artifacts` through the SELECT-only `sim.review` API.

## Implementation Locations

Primary implementation files:

- `sim/dynamics/model.py`: `OrbitalAttitudeDynamics.step`, orbit substepping,
  command acceleration application, and coupling to attitude/aero/SRP geometry.
- `sim/dynamics/orbit/propagator.py`: `OrbitPropagator`, perturbation plugin
  wiring, CR3BP dispatch, adaptive-step accounting, accelerated zonal RK4
  dispatch, and capability/profitability routing for fused and staged compiled
  force plans.
- `sim/dynamics/orbit/custom_force.py`: validation and stage-time adapter for
  opt-in external acceleration models.
- `sim/acceleration/kernels/orbit_force_plan.py`: exact compiled RK4 and
  per-component kernels shared by supported RK4, adaptive RKF78
  (`rkf78`/`adaptive`), and Dormand-Prince (`dopri5`) force plans.
- `sim/dynamics/orbit/accelerations.py`: two-body, J2, J3, J4, drag, lift,
  SRP, and third-body acceleration functions.
- `sim/dynamics/orbit/integrators.py`: RK4, RKF78, Dormand-Prince, and adaptive
  step-control helpers.
- `sim/dynamics/orbit/two_body.py`: standalone two-body RK4 helper used by
  selected estimation/control/mission modules.
- `sim/dynamics/orbit/spherical_harmonics.py`: spherical-harmonic coefficient
  parsing, frame handling, and perturbation acceleration.
- `sim/dynamics/orbit/atmosphere.py`: density-model dispatch used by drag and
  lift.
- `sim/dynamics/orbit/eclipse.py`: SRP Sun geometry and shadow factor.
- `sim/dynamics/orbit/cr3bp.py`: CR3BP system and Python RK4/adaptive
  (`rkf78`/`adaptive`/`dopri5`) propagation paths.
- `sim/runtime_support.py::_build_orbit_propagator`: scenario YAML to runtime
  propagator/plugin wiring.

`sim/dynamics/orbit/propagator.py` also contains a selected-planet
third-body plugin used by lower-level tests and extension paths. The ordinary
scenario builder currently wires the public Sun and Moon third-body booleans.

## Validation Evidence Hooks

Public implementation evidence includes targeted tests and validation harness
entries rather than a broad mission-assurance claim.

Useful commands:

```bash
.venv/bin/python -m pytest -q sim/tests/test_orbit_integrators.py
.venv/bin/python -m pytest -q sim/tests/test_orbit_j3_j4.py
.venv/bin/python -m pytest -q sim/tests/test_orbit_spherical_harmonics.py
.venv/bin/python -m pytest -q sim/tests/test_orbit_atmosphere_models.py
.venv/bin/python -m pytest -q sim/tests/test_orbit_planetary_third_body.py
```

Private validation harnesses cover orbit integrator selection/adaptive-step
behavior, epoch and eclipse behavior, atmosphere/drag/SRP/third-body/zonal
perturbation tests, and spherical-harmonic plugin wiring. The saved Orekit
cumulative J2/J3/J4 suite and optional HPOP and precise-orbit comparison workflows provide
stronger external-reference evidence for selected configured cases when the
required reference data and tooling are present.

For a completed run with review output enabled, reproducible evidence can be
queried with commands such as:

```bash
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, dt_s, samples FROM run_metadata"
.venv/bin/python -m sim.review outputs/<scenario_name> --query "SELECT time_s, object_id, pos_x_eci_km, pos_y_eci_km, pos_z_eci_km, vel_x_eci_km_s, vel_y_eci_km_s, vel_z_eci_km_s FROM object_state ORDER BY time_s, object_id LIMIT 20"
```

State the exact scenario, config fields, commit/worktree state, and review
query when using these outputs as evidence.

## Limitations And Non-Claims

The public orbit dynamics documentation does not claim:

- flight qualification,
- operational decision authority,
- validated accuracy for arbitrary mission envelopes,
- high-fidelity force-model accuracy for all orbit regimes,
- equivalence to STK, FreeFlyer, GMAT, Orekit, Basilisk, MATLAB/HPOP, or any
  program-specific truth model,
- correctness of user-supplied plugin modules, coefficients, space-weather
  inputs, TLEs, or scenario YAML.

Specific model limits:

- The default `two_body` path is a numerical propagation path, not a closed-form
  Kepler solver.
- Drag/lift accuracy depends on the selected atmosphere model and user-supplied
  mass, area, coefficient, attitude, and space-weather inputs.
- Public spherical-harmonic scenarios need inline terms or an explicit
  coefficient file. HPOP/GGM03 reference data is not bundled with the public
  core.
- OGP-SGP4 objects are passive catalog-style objects in the current
  implementation and do not accept OEL thrust or controllers.
- OGP-SGP4 support rejects TLEs with orbital period at or above 225 minutes
  because those cases require OGP-SDP4/deep-space/resonance handling.
- The optional acceleration path is a performance optimization for supported
  kernels; it does not change the validation envelope.
- Review-store evidence records what a completed run produced. It does not
  prove the selected force model is adequate for a mission-specific decision.

Users should independently validate initial conditions, force-model fidelity,
frame assumptions, numerical tolerances, environment inputs, controller and
actuator limits, and mission-specific safety/legal/compliance requirements.
