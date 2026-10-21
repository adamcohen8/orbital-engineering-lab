# Optional Earth precision forces

OEL's public ONP includes four experimental, opt-in Earth-centered force models. Each defaults
off and applies to special numerical propagation, not OGP or CR3BP. No Pro
entitlement is required. The `sim.pro_perturbations` package name is retained
for compatibility; these four models and their numeric helpers are public.

| Model key under `simulator.dynamics.orbit` | Physical scope | Options |
| --- | --- | --- |
| `earth_radiation` | Spherical-Earth Lambertian albedo and infrared pressure with Knocke seasonal climatology | `enabled`, `albedo`, `infrared`, `quadrature_order` (8–128; default 32) |
| `schwarzschild` | Earth-centered first post-Newtonian orbital acceleration | `enabled` |
| `solid_earth_tides` | Degree-2/3 anelastic response, induced degree 4, IERS 2010 frequency terms, permanent-tide convention and solid pole tide | `enabled`, required `tide_system` (`tide_free` or `zero_tide`), `pole_tide` |
| `ocean_tides` | Harmonic ocean tidal gravity and degree-2/order-1 ocean pole correction | `enabled`, `coeff_path`, `degree` (2–20), `order` (0–degree), `pole_tide` |

All four add acceleration to the selected Newtonian force stack. Direct solar
radiation pressure and Sun/Moon direct gravity remain separate choices. Solid
and ocean pole tides are distinct and can both be enabled. Pole tide defaults
on within an enabled tide section; ocean pole tide requires order at least 1.

## Run the demonstration

From a supported source checkout environment:

```bash
python run_simulation.py --config configs/precision_forces_demo.yaml --validate-only
python run_simulation.py --config configs/precision_forces_demo.yaml
python -m sim.review outputs/precision_forces_demo --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
```

The example runs a passive 600-second orbit with review evidence, plots and
attitude disabled. Its fixed EOP values and analytic Sun/Moon ephemerides are
demonstration inputs. The bundled tiny ocean coefficient file is OEL-authored
synthetic data that exercises the reader; it is not FES2004 or a physical ocean
model. Replace it with an authorized local fully normalized FES `Cnm-Snm` file
for a study requiring ocean tidal gravity. Height/amplitude/phase files are not
interchangeable. Expected columns are `DelC+ DelS+ DelC- DelS-`, in units of
10^-11. No model download or invented fallback occurs. Missing, malformed or
insufficient-degree files fail validation or construction.

Each successful run writes compact summary and standard review evidence in
`outputs/precision_forces_demo`. Inspect run identity before reusing its results.

## Inputs and limits

Radiation requires an absolute epoch and explicit ephemeris mode. Its response
is constant cannonball area, Cr and mass; it does not model spacecraft facets,
thermal recoil or torque. Albedo and IR remain independently inspectable.
Increasing quadrature order is a convergence study, not a different force
model. Stock Orekit Knocke has different visible-cap and emission-projection
geometry; timing or proximity to its trajectory is not a force-equivalence
claim.

Both tide models require an absolute epoch and EOP-backed `simulator.frames`,
using a covered trusted EOP file or explicit DUT1, TAI−UTC, xp and yp. Solid
tides also require explicit Sun/Moon ephemerides. Match the static gravity
field's permanent-tide convention explicitly; OEL does not infer it. Ocean
coefficients must not duplicate instantaneous corrections already in the
static field. IERS 2010 tidal coefficients do not imply an IERS 2010 frame
implementation: the supported scenario rotation remains IAU76/80 with EOP.
The original IERS 2010 piecewise mean-pole convention is retained. Updated
secular-pole conventions, atmospheric tides and station/loading displacement
are outside this implementation.

Schwarzschild supplies an additive weak-field, slow-motion correction in ECI.
Spin/frame dragging, third-body relativity and relativistic clock modeling are
outside its scope. These models support inspectable numerical studies, not
empirical orbit accuracy or flight qualification claims.

## Optional Rust execution

Set `simulator.dynamics.orbit.numeric_backend: rust` when the optional compatible
`oel-rust-orbit` extension is installed. Rust evaluates the same equations and
quadrature. Tide coefficient construction and the gradient can be evaluated
together in Rust; Python retains input loading, frames, ephemerides, validation
and the reference implementation. Older extensions lacking an optional new
force kernel retain the reference path.

Extensions advertising `ONP_PRECISION_FORCE_PLAN_VERSION = 1` also include all
four models in the ordered Rust ONP force plan. Select any subset, and combine
it with normalized spherical harmonics, drag, direct SRP, Sun/Moon gravity or
J2/J3/J4. Configured force order and each model's existing options are retained.
RK4, RKF78 and DOPRI5 use the prepared plan; supported passive segment/history
APIs can reuse it across steps. Python still prepares exact frames, ephemerides
and custom atmosphere inputs at each stage. Force arithmetic and integration
run through the native plan.

Older extensions, custom force subclasses and unsupported plugins use the
ordinary callback path. Explicit `numeric_backend: python` keeps the Python
reference implementation. The separate coupled mechanical/attitude and
forecast envelopes retain their existing eligibility rules; this extension
applies to ONP orbit propagation and its segment/history APIs.

Scientific table attribution and Apache-2.0 notices are retained in
`sim/pro_perturbations/NOTICE.txt` and `LICENSE-Orekit.txt`. External FES datasets
are not redistributed with OEL.

The public Rust source is a buildable, experimental kernel library at
[`rust/oel-precision-forces`](../rust/oel-precision-forces/README.md). It shares
OEL's authoritative force owners and keeps resource/frame/propagator policy
with the caller. Run its documented Cargo tests to check the synthetic native
contracts without any external FES dataset or private validation fixture.


## Explicit empirical RIC acceleration

`sim.pro_perturbations.empirical.EmpiricalRICAcceleration` is an opt-in Python
acceleration plugin with a source-public native force-plan implementation.
Its three constants and optional cosine/sine triples are radial, transverse
and orbit-normal accelerations in km/s². Harmonics use an explicit angular
frequency in rad/s and reference elapsed time in seconds; nonzero harmonic
coefficients require positive frequency. Transverse is the orbit-normal unit vector crossed with radial, and all
three directions are recomputed at each current integrator stage. A zero
coefficient does not add a physical perturbation. This model can represent
unmodeled accelerations for a declared study; it does not identify a thermal,
drag or other physical mechanism. Fit coefficients only on declared training
data, carry their uncertainty, and keep them fixed for forecast evaluation.
There is no automatic scenario enabling or qualification claim. Older native
extensions retain the reference callback unless their explicit empirical-RIC
capability is present. Harmonic execution additionally requires the explicit
harmonic capability; a constant-only native extension uses the reference
callback for harmonic inputs.
