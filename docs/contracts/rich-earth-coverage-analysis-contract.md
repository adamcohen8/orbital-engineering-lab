# Rich Earth Coverage Analysis Contract

Status: **frozen for Phase 3 programmatic implementation v0.1**.

Contract identifier: `oel.rich-earth-coverage-analysis.v0.1`.

This is an additive contract over Global Earth Coverage Analysis v0.1. It does
not revise the original axisymmetric-cone result or its semantic hashes. The
same deterministic state, attitude, WGS84, HEALPix NESTED, sampled-time,
sparse-interval, dwell, revisit, and censoring rules remain authoritative.

## Product Boundary

Rich Earth Coverage Analysis evaluates one body-mounted sensor against the
complete canonical Earth grid after propagation. It adds hard-edged
rectangular and pushbroom fields of view, target geometry and illumination
constraints, sampled footprint-boundary intersections, and a ground-track plus
footprint overlay.

This remains center-of-cell sampled coverage. Boundary curves are review
geometry and do not redefine cell membership or establish exact polygon area.
The analysis cannot alter the simulation and cannot be used as a global
runtime monitor.

## Normalized Programmatic Record

Every analysis declares:

- analysis, source-asset, state-provider, attitude-provider, and sensor IDs;
- achieved, replay, or explicitly analytic-ideal attitude provenance;
- HEALPix order and explicit strictly increasing analysis epochs;
- scalar-first `quat_body_from_sensor` mounting;
- exactly one supported FOV pattern;
- optional maximum slant range and service constraints;
- explicit Sun-position evidence and provider identity when illumination is
  constrained;
- boundary sampling count and resource limits.

The Phase 3 programmatic record uses radians and kilometres. Adapters may
accept degrees only when they normalize visibly before validation.

Unknown pattern kinds, unsupported constraints, silent attitude defaults, and
silent Sun models fail validation.

## Sensor Frame and Supported Patterns

The right-handed sensor frame uses `+Z` as boresight, `+X` as horizontal or
cross-track, and `+Y` as vertical or along-track. `quat_body_from_sensor` maps
sensor-frame vectors into the parent body frame.

Supported hard-edged patterns are:

1. `axisymmetric_hard_cone`
   - one half-angle strictly in `(0, pi/2)`;
   - membership uses the inclusive off-axis cone test from Global Earth
     Coverage Analysis v0.1.
2. `rectangular_hard_fov`
   - horizontal and vertical half-angles, each strictly in `(0, pi/2)`;
   - for target direction `(x_s, y_s, z_s)` in sensor coordinates, membership
     requires `z_s > 0`, `abs(atan2(x_s, z_s)) <= horizontal_half_angle`, and
     `abs(atan2(y_s, z_s)) <= vertical_half_angle`.
3. `pushbroom_hard_fov`
   - cross-track and along-track half-angles with the same bounded rectangular
     projection test;
   - the distinct kind preserves payload semantics and axis naming. It does
     not imply scanning, integration time, detector performance, or motion
     compensation.

All angular comparisons use the declared Phase 3 angular tolerance. The
pattern is a hard geometric gate, not an optical response model.

## Physical and Service Geometry

Physical line of sight uses the WGS84 ellipsoid and the same convex-surface
horizon predicate as the Phase 1 product. Exact tangency remains blocked.
Direct surface-target geometry calls require each target to satisfy the WGS84
ellipsoid equation within a `1e-12` dimensionless normalized-level tolerance
and each supplied normal to align with the normalized ellipsoid gradient within
a `1e-10` dot-product tolerance. These checks run before the Python or native
geometry kernel so an off-surface or inward-facing target cannot be treated as
valid evidence.

For each source/target pair:

- target off-nadir angle is between the observer-to-target direction and the
  observer's WGS84 geodetic-nadir direction;
- incidence angle is between the target outward ellipsoid normal and the
  target-to-observer direction;
- target Sun elevation is `asin(n dot u_sun)`, where `n` is the target outward
  normal and `u_sun` points from the target to the supplied Sun position.

Optional inclusive constraints are:

- maximum target off-nadir angle in `[0, pi/2]`;
- maximum incidence angle in `[0, pi/2]`;
- minimum and/or maximum target Sun elevation in `[-pi/2, pi/2]`, with minimum
  no greater than maximum.

If either illumination bound is enabled, a finite Sun ECI position is required
at every analysis epoch and `sun_provider_id` is mandatory. Phase 3 does not
silently generate Sun evidence.

Availability is the conjunction of line of sight, pattern, range, off-nadir,
incidence, and illumination gates. The deterministic primary-failure order is:

1. `earth_blocked`;
2. `outside_pattern`;
3. `outside_range`;
4. `off_nadir_exceeded`;
5. `incidence_exceeded`;
6. `illumination_rejected`.

An available target has `available` as its primary disposition. Counts by
primary disposition are retained per sample and in the summary. This reason
ledger is diagnostic; all enabled gates remain independently inspectable.

## Sampled Footprint Boundary

At every analysis epoch, Phase 3 generates an ordered set of boundary rays in
the sensor frame:

- a cone uses uniformly spaced azimuth around its angular boundary;
- a rectangular or pushbroom pattern samples each projected angular edge in
  clockwise order without duplicating corners.

Each ray is independently transformed through achieved or declared sensor
attitude and intersected with the WGS84 ellipsoid. The nearest positive
intersection is retained. A ray that does not hit Earth is a valid sampled
miss, not a fabricated limb point.

Per-sample boundary disposition is:

- `complete`: every boundary ray intersects Earth;
- `partial`: some but not all rays intersect Earth; or
- `no_intersection`: no boundary ray intersects Earth.

Boundary WGS84 geodetic latitude and normalized longitude are retained with a
hit mask. Antimeridian discontinuities are split for plotting. The boundary is
sampled review geometry: it is not an exact ellipsoid/cone silhouette, swept
footprint, polygon-overlap area, or alternative membership rule.

The source ground track uses the WGS84 geodetic subsatellite longitude and
latitude derived through the same ECI/ECEF frame path.

## Time, Metrics, and Regional Queries

Phase 3 uses the same independent sample epochs and left-closed, right-open
dwell accumulation as Phase 1. The final epoch is a snapshot with no following
duration. No between-sample access is inferred.

The rich result preserves the `GlobalCoverageResult` query surface: canonical
cell centers, sparse intervals, per-cell metrics, covered counts and fractions,
source interval semantic hash, and a global-Earth disposition. Phase 2 region
and point queries may therefore consume it without rerunning geometry.

Equal-area cell aggregation remains authoritative. Boundary polygons are never
used to calculate regional or global area.

## Artifacts

The stable Phase 3 products are:

- `rich_coverage_analysis_manifest.json`;
- `rich_coverage_summary.json`;
- `rich_coverage_samples.csv`;
- optional `rich_coverage_cells.csv`;
- `rich_coverage_intervals.npz`;
- `rich_coverage_footprints.npz`; and
- optional `rich_coverage_footprints.png` review overlay.

The interval NPZ uses the frozen Phase 1 sparse arrays. The footprint NPZ
contains sample times, subsatellite coordinates, boresight off-nadir angle,
boundary hit mask and coordinates, and boundary disposition codes. Arrays use
canonical ordering and explicit little-endian numeric types.

The semantic hash binds normalized scientific configuration, time samples,
sparse intervals, reason counts, ground track, boresight off-nadir evidence,
and sampled footprint-boundary arrays. Container and plot byte hashes are
separate because compression metadata and rendering are not scientific
identity.

## Resource and Failure Behavior

Preflight reports cells, samples, cell-time comparisons, boundary rays,
estimated working memory, and configured limits. The evaluator fails before
materialization when the declared envelope is exceeded.

Invalid state, attitude, mounting, FOV, constraint, epoch, Sun evidence,
canonical grid, or resource configuration fails closed. An observer on or
inside WGS84 is invalid. A partial or absent sampled boundary remains valid
evidence when the observer and rays are otherwise well formed.

Chunk size may affect performance but must not change cell membership,
intervals, metrics, reason counts, boundary arrays, or semantic hash.

## Acceptance Fixtures

Acceptance requires at minimum:

- rich-cone parity with the frozen Phase 1 evaluator when no new constraint is
  enabled;
- known rectangular and pushbroom inside/outside cases including corners;
- mounting and achieved-attitude rotation cases;
- nadir and limb off-nadir/incidence cases;
- day, night, and illumination-boundary cases using explicit Sun evidence;
- complete, partial, and no-intersection boundary cases;
- antimeridian and polar boundary/plot cases;
- chunk-size and artifact semantic-hash parity;
- Phase 2 region/point query compatibility;
- fail-closed malformed and resource-limit cases; and
- independent matched-assumption comparison before decision-grade claims.

## Explicit Non-Claims

Phase 3 v0.1 does not model or claim:

- exact footprint polygons, exact covered area, or swept coverage;
- scanning schedules, pushbroom integration, optical resolution, image quality,
  modulation transfer, smear, or detector performance;
- terrain, buildings, atmosphere, clouds, weather, or refraction;
- Sun occultation at the target beyond local WGS84 horizon elevation;
- gimbals, articulation limits, slew feasibility, tasking, or scheduling;
- communications or RF service;
- multiple-spacecraft union, multiplicity, or constellation availability;
- probabilistic availability or operational mission assurance.

Scenario YAML, automatic completed-run extraction, review-store tables, and
agent-facing tools remain adapters outside this scientific contract. The
implemented rich-coverage kernel, artifacts, plots, contract, and tests are
public core.
