# Planetary ONP

Planetary environments follow a **bring your own coefficients and atmosphere**
model. OEL provides the propagation framework, body identities, point-mass
constants, reference rotations and import contracts. Users supply compatible
gravity and atmosphere datasets for their studies. Mars includes built-in
GMM-2B gravity and a bounded MAVEN reference atmosphere as a ready-to-run
example; the same import contracts accept alternative Mars data.

Non-Earth primary bodies use a separate prepared Rust `PlanetaryPlan`. Omitting
`simulator.central_body`, or selecting `earth`, preserves existing normalized
Earth scenarios and Earth force dispatch. The new constructor is selected once;
Earth force codes, scalar layouts, atmosphere models and derivative routines
remain unchanged. This is deterministic serial passive trajectory propagation.

## Available envelope

| Primary | Point mass | Rotation | Harmonic gravity | Atmosphere |
| --- | --- | --- | --- | --- |
| Mars | DE440 physical center 499 GM | IAU 1991 gravity; IAU 2000 atmosphere | Built-in GMM-2B, degree 2–20, or user data | Built-in fixed MAVEN P00718 reference, 135.017–164.937 km, or user data |
| Moon, Mercury, Venus | Physical-center GM | IAU 2000 | User PDS SHA table and matching declared frame | None, bounded exponential or user table |
| Jupiter, Saturn, Uranus, Neptune | Physical-center GM | IAU 2000 | User PDS SHA table and matching declared frame | None, bounded exponential or user table |
| Sun | Physical-center GM, center 10 | IAU 2000 | Point mass only | None |

Definition availability does not qualify a real gravity or atmosphere dataset.
The additional-body tests exercise controlled lunar coefficients and synthetic
center-to-center ephemerides. They establish the extensible contract, not lunar
geophysical fidelity or ephemeris prediction. Giant reference radii do not imply
solid surfaces; their impact monitor defaults off, as does the Sun's.

GMM-3 deliberately raises an unavailable-model error: its Konopliv rotation must
be verified before admission. GMM-2B was selected as the documented initial
alternative. Its archive specifies IAU 1991. Do not attach an IAU 2000 or modern
IAU rotation to those coefficients. The GMM-2B radius 3397 km and GM
42828.371901284001 km³/s² apply coherently to both monopole and harmonics. They
are separate from the atmosphere's 3396.19/3376.20 km navigation ellipsoid and
from the default point-mass GM.

## Inspect a model before authoring

`sim.dynamics.planetary.preparation.planetary_model_catalog()` reports the
Schwarzschild setting path and whether the installed native extension supports
it. Select a profile by primary and gravity model to read the effective GM
(km³/s²), bundled rotation datasets, and their common inclusive date range.
For example, Mars point mass and Mars GMM-2B have different GMs.
Dates in the catalog and coverage errors are on the TDB calendar. The scenario
start epoch is still supplied in UTC. With custom frames or ephemerides,
validate the selected files to check their actual bounds. A rejected interval
identifies the limiting dataset and the requested and supported dates.

## Build and run

Follow [Building public native source](native-source-build.md) to install the
wheel that matches your checkout. The extension must export `PlanetaryPlan`,
`planetary_rotation`, `planetary_atmosphere_sample` and `planetary_tdb_seconds`.
Older Earth-compatible wheels fail planetary validation clearly. There is no
Python implementation or lower-fidelity fallback for the new force core.

```sh
python run_simulation.py --config configs/mars_reference_drag.yaml --safe-validate
python run_simulation.py --config configs/mars_reference_drag.yaml --validate-only
python run_simulation.py --config configs/mars_reference_drag.yaml
python run_simulation.py --config configs/sun_centered_srp.yaml --validate-only
python run_simulation.py --config configs/sun_centered_srp.yaml
```

Examples have distinct output directories. Existing output directories follow the
normal CLI archive/overwrite policy. Use a new output directory for each study.
For a long run use the documented durable `oel runs` lifecycle and retain its
accepted manifest and normalized-config digest; it uses the same engine.

The minimal configuration selects an absolute `initial_jd_utc`, primary body,
passive satellite, explicit mass and a Cartesian state:

```yaml
objects:
  probe:
    kind: satellite
    runtime_profile: trajectory_only
    specs: {mass_kg: 100}
    initial_state:
      position_inertial_km: [4000, 0, 0]
      velocity_inertial_km_s: [0, 3.27217, 0]
simulator:
  central_body: mars
  initial_jd_utc: 2457065.5
  duration_s: 120
  dt_s: 5
  planetary:
    gravity: {model: point_mass}
  dynamics:
    orbit: {integrator: rkf78, numeric_backend: rust}
```

Cartesian vectors are km and km/s in body-centered J2000; both vectors are
required. Elliptic COEs require explicit semimajor axis and
`reference_axes: J2000`; they use the selected gravity GM. Hyperbolic and
parabolic studies use Cartesian initialization. Existing COE aliases are
preserved. Earth ECI states, altitude-derived Earth orbit defaults and TLE/OGP
initialization are rejected. RK4, RKF78, DOPRI5 and the `adaptive` DOPRI5 alias
are available. `orbit_substep_s` limits segments when supplied; positive
`adaptive_atol`/`adaptive_rtol` retain existing component units km and km/s.

## Bring your own environment

The default for every non-Earth body, including Mars, is point-mass gravity
with no atmosphere. Harmonics and drag are explicit selections. The bundled
Mars example demonstrates the complete integration and provides a maintained
numerical reference; other bodies use user-supplied environmental data.

For example, this configuration fragment selects custom Venus gravity and
atmosphere files:

```yaml
simulator:
  central_body: venus
  planetary:
    gravity:
      model: pds_sha_static
      coeff_path: models/venus_gravity.sha
      label_path: models/venus_gravity.lbl
      frame_id: VENUS_MODEL_FIXED
      max_degree: 20
      max_order: 20
    gravity_frame:
      model: resource
      data_path: models/venus_gravity_frame.json
    atmosphere_frame: {model: iau2000}
    atmosphere:
      model: table
      data_path: models/venus_atmosphere.json
      out_of_domain: error
    drag: {enabled: true}
```

Supply the files before validating a complete scenario. The gravity-frame
resource must declare `VENUS_MODEL_FIXED` and reproduce the coefficient
dataset's actual convention; the atmosphere table in this example must declare
`IAU2000_VENUS`. Gravity coefficients carry their own GM and reference radius.
Atmosphere tables declare their altitude datum, density/wind units, conditions
and bounds. The formats are described below. This fragment does not establish
the fidelity of a particular Venus dataset.

Bring your own data currently means local static PDS SHA gravity coefficients
and bounded atmosphere tables or explicit exponential parameters. Proprietary
climatology packages, arbitrary provider code and unsupported file formats need
an explicit conversion or a separately supported adapter. OEL checks the input
contracts, records hashes and refuses incompatible selections or domain
violations; users remain responsible for dataset suitability and independent
validation of their chosen environment. Further built-in planetary gravity or
atmosphere datasets can be added as separately qualified resources.

## Gravity and frames

For simultaneous primary and external-body harmonics (including Earth and
Moon in one simulation), see [extended third-body gravity](additional-gravity.md).

```yaml
planetary:
  gravity: {model: gmm2b_static, max_degree: 20, max_order: 20}
  gravity_frame: {model: iau1991}
  atmosphere_frame: {model: iau2000}
```

Gravity uses fully normalized, no Condon–Shortley coefficients. The PDS SHA
importer validates eight header fields (radius, GM, GM uncertainty, maximum
degree/order, normalization flag, longitude and latitude offsets), followed by
six-column `n,m,C,S,sigma_C,sigma_S` records. It rejects duplicates, missing
requested terms, nonzero degree one, non-unit degree zero, unknown
normalization, nonzero reference offsets, invalid uncertainties and nonfinite
data. Omitted degree zero means unit monopole and omitted degree one means
center of mass; only degree 2+ enters the harmonic perturbation. Harmonic
stages reject positions at or below the dataset reference sphere.

Use `pds_sha_static` for another declared static model, supplying `coeff_path`,
`label_path`, exact `frame_id`, degree/order, and optional coefficient `sha256`.
The detached label must identify the selected body. This bounded importer
accepts detached numeric SHA files; attached labels or binary/covariance
products must be converted transparently before use. An authored `frame_id`
is an assertion that the resource matches the dataset's published convention,
not automatic qualification of an unknown archive.

A `gravity_frame` or `atmosphere_frame` with `model: resource` accepts a local
JSON resource with body ID, frame ID, J2000 axes, source, TDB coverage and either
`pole_meridian` or `euler_series` parameters. Pole RA/DEC polynomials use Julian
centuries from J2000; meridian polynomials use TDB days. Euler resource rates
use degrees/day, with optional periodic sine/cosine terms. See bundled JSON
resources for the complete schema. Native rotation includes analytic derivatives
and quadratic/periodic terms (including lunar and Neptune terms).

Elapsed time is TAI seconds, anchored once from UTC. TDB uses a low-order TT–TDB
approximation with its derivative chain rule, not exact SPICE ET conversion.
The archived IAU 1991/2000 definitions are admitted only inside the declared
1972–2030 implementation envelope. This is not a long-term orientation accuracy
claim. Frames use `r_B=R r_I`, `v_B=R v_I+dR/dt r_I`; native regular solid
harmonics avoid longitude/latitude singularities at the poles.

## Passive attitude around a planetary primary

Every supported primary can advance a rigid body's orientation and spin alongside its
orbit. Enable `simulator.dynamics.attitude.enabled` and supply the object's body-frame
inertia about its mass center. For example, these sections enable gravity-gradient
torque on a Saturn-centered object (moments in kg m²; initial spin in rad/s).

```yaml
simulator:
  central_body: saturn
  dynamics:
    attitude:
      enabled: true
      attitude_substep_s: 5
      disturbance_torques: {gravity_gradient: true}
objects:
  probe:
    specs:
      mass_properties:
        inertia_reference_point: center_of_mass
        inertia_kg_m2: [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
    initial_state:
      attitude_quat_bn: [1, 0, 0, 0]
      angular_rate_body_rad_s: [0, 0, 0.001]
```

The scalar-first quaternion maps J2000 inertial vectors into spacecraft body axes.
Spin components are in spacecraft body axes. OEL evaluates Euler's rigid-body
equations at RK4 stages synchronized with the native orbit. Use an attitude
substep short enough to resolve both spin and orbital forcing and verify
convergence by reducing it. All disturbance torques default to off.

| Disturbance switch | Inputs and physical model |
| --- | --- |
| `gravity_gradient` | Monopole tide using the selected gravity dataset's effective GM, the instantaneous primary-relative position, and the full inertia tensor. Higher-degree gravity-gradient torques are outside this model. |
| `drag` | Enable planetary drag and supply `specs.drag_cp_offset_body_m`. Uses the selected atmosphere, wind and rotating frame through the orbit model's drag force. |
| `srp` | Enable planetary SRP and supply `specs.srp_cp_offset_body_m`. Uses the orbit model's Sun direction, distance and selected eclipse model. In a heliocentric case the Sun is the origin. |
| `magnetic` | Supply `specs.magnetic_dipole_body_a_m2` and `simulator.planetary.magnetic_field` with `model: centered_dipole`, the matching `body_id`, gravity rotation `frame_id`, and a three-component `dipole_t_m3` in that primary-fixed frame. The field rotates with the primary and falls as inverse radius cubed. Supply coefficients appropriate to the body and epoch. |

Pressure offsets are vectors **from the spacecraft mass center to the center
of pressure** in body-axis meters. The pressure forces keep the planetary
model's constant effective areas. The magnetic field is an explicitly supplied
idealized centered dipole. A measured field map or space-weather model would
require a different field provider.

A standard review store records `planetary_attitude` for enabled runs. Join
it to `planetary_state` on `(object_id, sample_index)` for simultaneous orbit,
quaternion, spin, and individual instantaneous torque vectors in N m. Compact
review omits sample rows. With attitude disabled, the initial orientation and
spin in the full state history stay fixed, and there are no propagated attitude rows.

## Atmosphere and drag

The default atmosphere is `none`. Drag requires an explicit atmosphere and
positive constant `cd` and `drag_area_m2`; mass is fixed.
The planetary path requires one fixed total `specs.mass_kg`; it rejects
`dry_mass_kg` and `fuel_mass_kg` instead of silently ignoring them.
Wind is Cartesian m/s in the selected rotating body frame. Rotation is included once through `dR/dt`;
relative flow is `R.T @ (R @ v + dR/dt @ r - wind/1000)`. Native drag converts
m/s² to km/s² explicitly. Attitude/projected-area coupling, lift, gas-surface
calibration, heating and entry physics are outside this envelope.

`mars_reference_table` is one inbound MAVEN accelerometer profile, P00718 on
2015-02-12. The bundled original table and XML label are content-bound sources.
OEL selects 135–165 km inbound samples with positive density exceeding three
reported standard deviations; the resulting actual bounds are
135.017–164.937 km. It converts kg/km³ to kg/m³ and linearly interpolates log
density; Cartesian winds interpolate linearly. Sample values are reproduced.
Altitude is geodetic height over the documented 3396.19/3376.20 km oblate
navigation ellipsoid. Wind is assumed zero relative to IAU 2000 rotation;
it was not measured. Temperature and pressure remain unavailable.

This is a fixed reference experiment. Latitude, longitude, local solar time,
solar zenith angle, reference epoch, selection and uncertainty notes are saved
in provenance. Along-track gradients are folded into the profile. It is not a
global, seasonal or weather model; archived sigma is not total uncertainty.
The archive warns of calibration biases around 10% in deep dips and 20% in
science orbits. Do not interpret numerical agreement as density prediction.

Atmosphere options are specific to the selected model. The built-in
`mars_reference_table` permits an optional `sha256` pin for the bundled JSON,
but rejects a custom path or inline density/datum/wind overrides. Use `table`
for custom data or `exponential` for inline parameters. An accepted hash pin
must match the actual selected resource.

A custom `table` requires local JSON `data_path`, optional `sha256`, `body_id`,
`frame_id`, `source`, `conditions`, density units `kg/m3`, wind units `m/s`,
`wind_frame: body_fixed_cartesian`, ordered `altitude_km`, positive
`density_kg_m3`, an N×3 `wind_body_m_s` array, and `altitude_domain_km` equal to
its endpoints. Its `altitude_datum` is `radial_sphere` with `radius_km`, or
`radial_ellipsoid`/`geodetic_ellipsoid` with `equatorial_radius_km` and
`polar_radius_km`. Custom tables can later carry climatology selected upstream;
the current evaluation is a fixed one-dimensional table.

`exponential` requires the same explicit datum and bounded altitude domain,
positive reference density and scale height, reference altitude, and optional
constant wind. It is a controlled experiment, not a Mars climatology.

Outside the domain the default is an error at the offending integration stage.
Explicit `out_of_domain: vacuum_above` sets density to zero above the table and
records `within_atmosphere_domain=false`; it never extrapolates below the
bottom. That hard cutoff can affect numerical convergence. A 1e-9 km endpoint
allowance only handles floating point reconstruction of exact table endpoints.

## Ephemerides, SRP and other bodies

Third bodies and non-Sun SRP require an explicit local geometric ephemeris JSON
table. There is no Earth analytic fallback or implicit barycenter substitution:

```yaml
planetary:
  third_bodies:
    - {target_center_id: 10, mu_km3_s2: 132712440041.27942, name: Sun}
  srp: {enabled: true, shadow_model: cylindrical}
  ephemeris: {model: table, data_path: mars_geometric_ephemeris.json}
```

The table declares `observer_center_id` equal to the selected physical origin,
`inertial_axes: J2000`, `aberration: NONE`, `time_scale: TDB`, position/velocity
units `km`/`km/s`, `source`, ordered `tdb_seconds_past_j2000`, and
`states_by_target_center` mapping integer NAIF IDs to N×6 position/velocity
rows. Cubic Hermite interpolation uses both positions and velocities and must
cover every stage in the study. Target identities and external GMs are explicit;
a system barycenter and a physical center are different models. Users must
qualify supplied tables and their sampling cadence against their reference.

External gravity is differential acceleration relative to the primary origin.
Cannonball SRP uses 1361 W/m² at 1 au and inverse-square spacecraft–Sun distance,
with positive fixed `cr` and `srp_area_m2`. Shadow choices are `none` or a primary
reference-sphere cylindrical model. Sun-centered SRP uses a source at zero;
the Sun cannot eclipse itself. Solar harmonics, neutral drag, corona/plasma,
and other occulting bodies are unsupported.

Set `simulator.dynamics.orbit.schwarzschild.enabled: true` to add the leading
weak-field relativistic acceleration of the selected primary. Every supported
primary uses the same setting, which defaults off. Each integrator stage uses
the primary-relative inertial position and velocity and the GM of the selected
gravity model, including its dataset GM when using harmonics. The planetary
review force breakdown records the contribution as `schwarzschild`. Spin,
relativistic third-body interactions and clock effects are outside this model.

## Evidence and compatibility

Standard review stores neutral coordinates in `planetary_state` and origin,
axes, hashes and model definitions in `planetary_environment`. Existing Earth
`object_state` tables remain empty for these runs. Sample diagnostics expose
altitude, density, relative speed, domain status and selected force vectors;
these are recorded sample points, not all adaptive stages. Compact review omits
state histories. JSON, CSV, NPZ, public records and snapshots carry explicit
origin/axes. Internal legacy `StateTruth` ECI field names are storage seams.

```sh
python -m sim.review outputs/mars_reference_drag --query "SELECT time_s, altitude_km, density_kg_m3, relative_speed_m_s FROM planetary_state ORDER BY time_s" --json
```

Use `EvidencePlotter(...).recipe(...)` with `planetary_altitude_km`,
`planetary_density_kg_m3`, `planetary_relative_speed_m_s`,
`planetary_inertial_xy` or `planetary_drag`. Series identify the primary body.
Earth recipes reject planetary stores. Public API relative vectors require
`frame='inertial'`; range/closest-approach queries use neutral vectors.
Earth RIC and ECI recipes are unavailable. ECI-only state-product interchange
rejects planetary stores instead of relabeling them.

Planetary configuration rejects Earth environment/frame/termination options,
OGP/TLE, attitude control, FSW, controllers/estimators, ground access/coverage,
resources/power/thermal, rockets/reentry diagnostics, plugins, collisions,
parallel object stepping and campaigns. No MCP tools are added. Impact is a
sampled endpoint check against an explicit reference sphere (default nominal
radius for solid bodies), producing `central_body_impact`. It is not terrain,
ellipsoid collision, a continuous crossing solver or chord-based detection.
If a primary harmonic force stage or its integrated endpoint would enter the
gravity reference sphere first, the run stops at the last completed sample
with `primary_gravity_domain_exit`. The failed serial interval is discarded
for every object. Summaries record the attempted interval end, reference
radius and affected object; this event is not physical impact. The engine
never evaluates an interior harmonic field or fabricates a boundary state.
Hosted execution and AI output workflows reject non-Earth configurations until
their complete origin/frame contracts are qualified.

Resources are loaded locally before propagation, hashed and checked before
outer steps and publication. Preparation does not execute provider code or
network requests. Configuration path policy still applies. Changes after
preparation fail instead of silently altering the force plan.

## Reproducible numerical qualification

```sh
python -m pytest sim/tests/test_planetary_onp.py
python rust/oel-orbit/benchmarks/qualify_planetary.py \
  --pck2000 /path/to/pck00008.tpc \
  --pck1991-source /path/to/pck00005.tpc --output planetary_qualification.json
```

The second command requires SciPy and SpiceyPy only as reference tools. It checks
source hashes and compares rotations/derivatives with CSPICE, harmonics with an
independent associated-Legendre potential gradient, and the combined 20×20
reference-atmosphere trajectory with SciPy DOP853 over 120 seconds. Frozen gates
are 3e-11 km/s² force component error, 3e-11 rotation matrix error,
3e-15 s⁻¹ derivative error, 3 mm position and 3e-6 m/s velocity error.
The test suite additionally checks Kepler conservation/trajectories, poles,
wind/drag units, bounded interpolation, repeatability, step refinement,
resource tamper, origin mismatch and unsupported consumers.

These checks support bounded numerical agreement. They do not establish modern
Mars gravity fidelity, atmosphere prediction, a complete orbit-decay model,
mission observation agreement or flight qualification. Earth timing is measured
separately using matched baseline and final inputs and deterministic hashes.

Data sources and transformation rules are in
[the planetary data inventory](../sim/dynamics/planetary/data/README.md).
