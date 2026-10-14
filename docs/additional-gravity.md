# Extended third-body gravity

ONP can combine primary-body gravity with the spherical harmonic gravity of
additional bodies in the same passive simulation. An Earth-centered run keeps
the existing Earth gravity selection and adds, for example, a lunar field.
A Moon-centered run can use lunar primary harmonics and add Earth's field.
Mars and the other supported planetary origins, including the Sun, can use
the same optional external-field path. Solar harmonics are unsupported.

This feature requires `oel-rust-orbit` >=0.18.2,<0.19. The physics and integration
are implemented in public Rust source. Coefficients, rotation definitions and
geometric ephemerides are prepared once through declarative Python adapters.
Other bodies remain bring your own coefficients; Mars GMM-2B is the built-in
example. The feature does not bundle or qualify a lunar gravity dataset.
See [native source builds](native-source-build.md) for the public crate build
and test commands.

## Earth-centered selection

Add this section under `simulator` alongside the existing Earth settings:

```yaml
initial_jd_utc: 2459669.5
additional_gravity:
  ephemeris:
    model: table
    data_path: data/earth_origin_ephemeris.json
  sources:
    - body_id: moon
      gravity:
        model: pds_sha_static
        coeff_path: data/moon.sha
        label_path: data/moon.lbl
        frame_id: YOUR_EXACT_LUNAR_MODEL_FRAME
        max_degree: 20
        max_order: 20
      gravity_frame:
        model: resource
        data_path: data/moon_rotation.json
dynamics:
  orbit:
    model: two_body
    propagation_method: special
    numeric_backend: rust
    integrator: rkf78
    third_body_moon: false
    spherical_harmonics:
      enabled: true
      source: icgem
      coeff_path: data/earth.gfc
      degree: 20
      order: 20
      frame_model: iau76_80_eop
      eop_path: data/eop.txt
  attitude: {enabled: false}
```

Each satellite must use `runtime_profile: trajectory_only` and a passive ONP
initial state. Earth states retain `position_eci_km`/`velocity_eci_km_s` or their
existing COEs. Non-Earth states retain the explicit J2000 planetary contract.
Select additional-body degree and order independently from the primary field.
The additional field currently supports degree 2–20 and order 0–degree. No
distance-dependent truncation is performed.

The [planetary gravity, rotation and ephemeris contracts](planetary-onp.md)
apply to each additional source. An ephemeris table must use the simulation's
physical center as observer (399 for Earth, 301 for the Moon), geometric `NONE`
states, J2000 axes, km and km/s, and TDB seconds past J2000. It must cover the
entire declared interval, with position and velocity supplied for each target.
Native cubic Hermite interpolation evaluates each integration stage. Time
conversion retains the planetary path's low-order TT–TDB approximation.

A coefficient frame must match the selected rotation's exact frame identity.
The bundled archived `iau2000` Moon frame is available, but should be selected
only when it matches the gravity dataset's published convention. A modern
lunar PA/ME model may require a different rotation resource; changing the
frame label does not convert its coefficients or qualify that frame.

## Earth as an external field

For a Moon-centered simulation, retain lunar primary settings under
`simulator.planetary` and use this additional source:

```yaml
additional_gravity:
  ephemeris: {model: table, data_path: data/moon_origin_ephemeris.json}
  sources:
    - body_id: earth
      gravity:
        model: earth_static
        coeff_path: data/earth.gfc
        frame_id: YOUR_EARTH_MODEL_FRAME
        max_degree: 20
        max_order: 20
      gravity_frame:
        model: earth
        frame_id: YOUR_EARTH_MODEL_FRAME
        frame_model: iau76_80_eop
        eop_path: data/eop.txt
```

`earth_static` accepts static ICGEM `gfc` records with explicit
`norm fully_normalized`, positive `earth_gravity_constant` in m³/s² and
`radius` in meters. Dataset GM and radius govern both its monopole and its
harmonics. Time-variable records, duplicate or missing requested terms,
nonfinite coefficients, non-unit degree zero and nonzero degree one are
rejected.
ICGEM keywords are handled case-insensitively by one validation/ingestion
pass. Duplicate coefficients and duplicate normalization, GM or radius
headers are rejected even when their casing differs.
The native frame adapter reuses the existing Earth IAU76/80 EOP evaluator
with extrapolation disabled. It does not route Earth primary
propagation through the planetary engine. Earth may alternatively use static
PDS SHA coefficients and an explicit `resource` rotation.

## Relative acceleration and protections

For source body B at center-to-center position **b**, spacecraft position **r**,
and full inertial-frame gravity field **gB** (monopole plus harmonics), ONP adds:

`aB(r,t) = gB(r − b,t) − gB(−b,t)`.

Both terms use the same coefficient GM, radius and body-fixed orientation.
Subtracting the field at the origin includes the origin's acceleration and
ensures this contribution vanishes at r = 0. This follows the relative-frame
convention of [third-body attraction](https://www.orekit.org/site-orekit-13.1.5/apidocs/org/orekit/forces/gravity/ThirdBodyAttraction.html),
extended here to the complete static harmonic field.

The monopole is already included. Selecting the same body through Earth
`third_body_moon` or planetary `third_bodies` is an error. Self sources,
duplicate sources, mismatched origins/frames, insufficient table/frame/EOP
coverage and resource hash changes are rejected. Spacecraft and simulation
origin must both remain outside every additional source's gravity reference
sphere. Optional file SHA256 pins are supported, and every resource is hashed
and recorded even without a supplied pin.

The initial supported envelope is passive, serial ONP with RK4, RKF78 or
DOPRI5. Controllers, attitude coupling, TLE/OGP, CR3BP, custom/system force
plugins, covariance/campaign studies and reentry are outside its qualified
envelope. Additional fields are opt-in; ordinary Earth configurations retain
their serialized form, exact propagator type, force ABI and existing native
integration/history loops. The opt-in subclass disables speculative kernels
that do not include the added field and uses ordinary lifecycle stepping.

## Evidence and model limits

Earth run summaries include per-object `additional_gravity` provenance.
Standard/full review stores add `additional_gravity_environment` and
`additional_gravity_acceleration`, with sample index, time, source and inertial
acceleration in km/s². Ordinary Earth review schemas are unchanged. Planetary
review `forces_json` includes `additional_gravity_<center_id>` alongside its
primary forces, and planetary environment provenance includes these sources.

Tests compare rotated complete fields to an independent associated-Legendre
potential gradient, exercise origin cancellation and duplicate/resource
guards, and compare trajectories with analytic J2 fields integrated by SciPy
DOP853. These synthetic fixtures establish implementation correctness within
their tolerances; they do not validate a user-supplied ephemeris, coefficient
model or frame for a particular mission. Ephemeris interpolation accuracy,
orientation accuracy, harmonic truncation and exterior-field validity remain
study-specific requirements.
