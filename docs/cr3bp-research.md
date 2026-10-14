# CR3BP research workflows

The experimental research surface extends the Earth–Moon CR3BP used by RPO
Trainer with coordinate transformations, Jacobi diagnostics, zero-velocity
slices, and review-backed figures/movies. It is an ideal circular-primary model,
not an ephemeris-aligned lunar mission-design or operational navigation product.
It does not provide differential correction, family continuation, or manifolds.

## Coordinate contract

`sim.dynamics.orbit.transform_cr3bp_state` accepts a physical state `[..., 6]`
(position km, velocity km/s) and scalar or broadcastable times in seconds.
Source and destination axes independently select `rotating` or `inertial`;
origins independently select `barycenter`, `p1`, or `p2` (Earth and Moon for the
named Earth–Moon system). Axes are right handed. Rotating x points from P1 to P2;
z is normal to the primaries' orbit, and rotation is positive about z.

The rotation angle is `reference_angle_rad + n * (time_s-reference_time_s)`.
Defaults align rotating and nonrotating axes at run time zero. This is a model
orientation convention, **not J2000, ICRF, or real ephemeris alignment**.
Primary-centered inertial axes do not rotate, but their origins accelerate.
Velocity is the derivative of position relative to the chosen origin in the
chosen axes. The transformation includes both origin motion and `omega × r`.
It does not transform accelerations, attitudes, or covariances.

```python
from sim.dynamics.orbit import transform_cr3bp_state

moon_inertial = transform_cr3bp_state(
    rotating_barycentric_states, times_s,
    target_axes="inertial", target_origin="p2",
    reference_time_s=0.0, reference_angle_rad=0.0,
)
original = transform_cr3bp_state(
    moon_inertial, times_s,
    source_axes="inertial", source_origin="p2",
    target_axes="rotating", target_origin="barycenter",
)
```

Existing `cr3bp_dimensional_state` / `cr3bp_nondimensional_state` in
`sim.dynamics.orbit.cr3bp` convert individual states using the system length
and velocity scales. Research transforms accept physical units only.
The named system selector supports Earth–Moon; low-level research helpers also
accept explicit `CR3BPSystem` parameters, which do not establish qualification
for another system or make Earth–Moon seed states valid in that system.

## Invariants and zero-velocity geometry

`cr3bp_jacobi_constant` accepts physical **barycentric rotating** states and
returns dimensionless `C = 2 Omega - |v_nd|²`. `cr3bp_jacobi_diagnostics` reports
initial/final C and maximum change relative to the first sample. Conservation
is expected only for unforced ideal CR3BP; commanded thrust can change C.
Do not classify a driven run's change as integrator error.

`cr3bp_libration_points` returns all five equilibrium positions in rotating
barycentric km. `cr3bp_zero_velocity_grid` returns plane coordinates and
`2 Omega - C`; negative values are forbidden on the selected slice. Primary
singularities have infinite potential. Supported planes are xy, xz, and yz.
Grid resolution is bounded to 16–1000 points per dimension.

A contour for a spatial orbit is a **slice of the zero-velocity surface**, not
an assertion that its projected trajectory must remain in the slice's allowed
region. By default, the plot's C is fixed at the first recorded sample unless supplied
explicitly. Movies can instead select the instantaneous mode described below. Under thrust it is a reference contour, not a conserved boundary.

## Completed-run plots and movies

Enable `outputs.review.enabled: true` before validating and executing a study.
Use `model: cr3bp`, `cr3bp_system: earth_moon`, explicit rotating state or halo
seed initialization, and disable Earth-impact termination for barycentric
cislunar studies. A ready-to-validate unforced example is `configs/cr3bp_research.yaml`. Validate
it with `python run_simulation.py --config configs/cr3bp_research.yaml --validate-only`
and run the same command without `--validate-only`, using a fresh output directory.

```bash
python -m sim.review cr3bp outputs/my_run --object vehicle --kind trajectory
python -m sim.review cr3bp outputs/my_run --object vehicle --kind trajectory --axes inertial --origin p2
python -m sim.review cr3bp outputs/my_run --object vehicle --kind zero_velocity --plane xy --slice-km 0
python -m sim.review cr3bp outputs/my_run --object vehicle --kind jacobi
python -m sim.review cr3bp outputs/my_run --object vehicle --kind trajectory --movie-format mp4 --frames 120 --fps 20
```

Use `--bounds-km UMIN UMAX VMIN VMAX` for a local view, `--jacobi C` for an
explicit contour, and `--style oel_light` for a light-background alternative to the default OEL dark theme. `--plane xz` or `yz`
selects another projection. Movies also support GIF and rotating zero-velocity
overlays. Rendering transforms recorded samples; it never reruns dynamics.

Python callers use `sim.review.render_cr3bp_research(workspace, object_id, ...)`.
Local MCP callers use the dedicated `oel.plan_cr3bp_render.v1` and
`oel.render_cr3bp.v1` contracts, rather than a generic plot recipe ID. Plan
against the recorded evidence, then render the unchanged specification with
the matching plan ID and operator-configured write approval. The MCP surface
preserves this renderer's experimental model and frame limits; its declared
resource bounds may be narrower than the Python/CLI interface.

The reader requires explicit `cr3bp_rotating` frame and matching CR3BP system
metadata. Missing frames, ECI input, unknown systems, duplicate/nonfinite times,
and truncated histories fail closed. Legacy `*_eci_*` review column names are
compatibility names only; recorded frame metadata controls interpretation.

Each output has a JSON receipt with query/parameters, source-store identity,
system parameters, frame/origin/units/orientation, C reference, diagnostics,
plot quality results, and artifact hashes. Entries are registered in
`review/generated_artifacts.json`. Movies include a deterministic contact sheet,
frame timestamps, and sampling convention. They select nearest recorded states
at uniform requested times, without interpolation; coarse source sampling may
produce held frames. Primaries and spacecraft use the same selected time.

## Agent presentation and verification

- Choose frame, origin, plane, and bounds to answer the actual question. Show
  complementary projections for spatial motion; never imply a 2D projection is
  a planar orbit. Use equal spatial axis scales and stable limits in movies.
- State that zero-velocity shading applies only to the labeled plane/slice;
  record whether C came from the first sample or an explicit value.
- Inspect the PNG, movie, and contact sheet. Check labels, contours, timestamps,
  aspect ratio, cropping, contrast, and visible motion. Automated quality
  receipts leave visual QA pending and do not establish presentation readiness.
- Explain conservation only when the run is unforced. Compare step sizes before
  attributing change to numerical error. Do not infer external physical accuracy
  from invariant conservation or periodic closure alone.

`sim/tests/test_cr3bp_research.py` verifies frame/origin round trips, velocity
transport by position finite differences, equilibria, Jacobi convergence, and
closure of the two corrected seeds with independent adaptive integration.
The original small training seed is not claimed to be a corrected periodic
orbit. These are numerical verification checks, not empirical qualification.

Numeric system parameters are resolved from this installation's maintained named-system
constants and recorded in the receipt; legacy review stores do not retain those
numeric constants. Requalify old evidence if the system definition has changed.

## Adaptive CR3BP integration

Runtime CR3BP propagation honors `simulator.dynamics.orbit.integrator`:
`rk4` (unchanged default), `rkf78`, `adaptive` (RKF78 alias), or `dopri5`.
For example:

```yaml
simulator:
  dynamics:
    orbit:
      model: cr3bp
      cr3bp_system: earth_moon
      integrator: rkf78
      adaptive_atol: 1.0e-12
      adaptive_rtol: 1.0e-10
```

Adaptive steps stay inside the requested outer/orbit-substep interval; output
sampling and command update times remain unchanged. A commanded acceleration
is held constant over that propagation call. The runtime retains the suggested
next internal step and exposes accepted/rejected step counts through
`OrbitPropagator.last_adaptive_step_info` and `adaptive_step_info`.

Both `propagate_cr3bp_state` and `propagate_cr3bp_reference_stm` in
`sim.dynamics.orbit.cr3bp` accept `integrator`, `adaptive_atol`, `adaptive_rtol`,
`h_init`, and `return_info`. Defaults preserve existing return shapes.
They also accept `numeric_backend="rust"` with the CR3BP-enabled
`oel_rust_orbit` wheel (0.6.0 or newer); Rust is the default. Scenarios
select it with `simulator.dynamics.orbit.numeric_backend: rust` alongside
`model: cr3bp`. The native backend preserves rotating barycentric units and
the configured system parameters, and evaluates the reference Jacobian/STM
in Rust without Python stage callbacks. Older wheels are rejected explicitly.
With `return_info=True`, state propagation returns `(state, info)` and STM
propagation returns `(reference, stm, info)`; RK4's info is `None`.
The augmented 42-component reference/STM system uses shared adaptive steps.
Absolute tolerances apply componentwise in physical state/STM units; relative
tolerance is dimensionless. This is error control, not a global accuracy bound.

Trainer predictions and halo-seed phase initialization still use their existing
RK4 defaults unless their direct propagation calls explicitly choose otherwise.
Selecting a runtime integrator does not retroactively change a phased seed.
The research example selects RKF78; Trainer configurations remain unchanged.

`sim/tests/test_cr3bp_integrators.py` checks dispatch/accounting, RK4 parity,
NRHO tolerance convergence against independent DOP853 integration, driven-state
accuracy, and STM finite differences. This is numerical verification within
ideal CR3BP, not ephemeris or observational validation.
`sim/tests/test_rust_cr3bp_backend.py` additionally checks native/Python state
and STM parity, DOP853 integration, Jacobi conservation, invalid inputs,
runtime selection and adaptive restart evidence.

## Maneuver-responsive zero-velocity animations

Use `--jacobi-mode instantaneous` to recompute C from the selected spacecraft's
recorded barycentric rotating state at each movie sample. The potential mesh is
computed once; the contour and forbidden-region shading are redrawn at C(t).

```bash
python -m sim.review cr3bp outputs/my_run --object vehicle \
  --kind zero_velocity --jacobi-mode instantaneous --movie-format mp4 \
  --plane xy --slice-km 0 --frames 120 --fps 20
```

Python uses the same `jacobi_mode="instantaneous"` option. It requires a
zero-velocity movie with rotating axes and cannot be combined with a fixed
`jacobi` override. The default remains `fixed` for backward compatibility.
The companion static PNG is the first-sample reference view, not a picture of
all changing energy boundaries.

During a coast, C and the contour should remain stable to numerical accuracy;
during a finite burn they evolve. This is an **instantaneous coast-accessibility
boundary**: continuing to thrust can change C and the accessible region. For
spatial motion the shaded region applies only to the labeled slice; the
trajectory is projected onto it. Do not interpret the shading as a permanent
barrier for a driven spacecraft or for an off-slice trajectory.

The movie title displays C(t). The receipt records mode, selected sample
indices, timestamps, and per-frame Jacobi values. Contours, shading, spacecraft,
and time labels use the same recorded sample. No dynamics or intermediate burn
states are invented by the renderer. A short burn or impulsive jump can only be
shown to the resolution of the saved samples and chosen movie frames; increase
recording/movie cadence when maneuver timing matters. Duplicate-time pre/post
states remain unsupported by this strictly increasing-time review reader.
