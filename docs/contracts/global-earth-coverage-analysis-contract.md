# Global Earth Coverage Analysis Contract

Status: **frozen for Global Earth Coverage Analysis v0.2 design and
implementation**.

Implementation status: **Phase 1 and Phase 2 programmatic kernels
implemented**. The additive implementation lives in `sim.analysis.healpix`,
`sim.analysis.observer_target_geometry`, and
`sim.analysis.global_coverage`, with region-mask and point-cell queries in
`sim.analysis.coverage_queries`. It accepts explicit deterministic state and
attitude arrays, performs resource preflight, content-binds the input arrays
and frame provenance, evaluates the frozen global cell-center predicate,
builds sparse intervals and metrics, aggregates
versioned declared regions without rerunning geometry, maps WGS84 points with
the canonical HEALPix rule, and writes deterministic query evidence bound to
the source interval hash. Canonical conical coverage has evidence-only
scenario, completed ONP/review, ECI OGP, and primary review-store adapters.
The canonical conical/directed-link workflow also has a supported
`coverage_link_review` agent recipe, named saved queries, and content-bound
plot recipes. Rich footprints, communications coverage, aggregation, tasking,
and a concrete causal ONP/FSW consumer do not yet have equivalent public agent
adapters.

Rich rectangular/pushbroom FOVs, explicit service constraints, and sampled
footprint-boundary review geometry are governed separately by
[`Rich Earth Coverage Analysis v0.1`](rich-earth-coverage-analysis-contract.md);
they do not revise this contract's Phase 1 conical semantics.

This document freezes the initial scientific and artifact contract. A behavior
change that contradicts it requires an explicit contract revision; additive
fields must remain backward compatible.

## Product Boundary

Global Earth Coverage Analysis v0.2 evaluates where and when one
spacecraft-mounted conical sensor covers representative points spanning the
complete Earth surface. The global product is authoritative. Regions and
points are queries over that same global cell identity, not different coverage
models.

Coverage is post-processed from a declared state and attitude provider. It is
not evaluated inside an orbit integrator and it cannot alter simulated
behavior. Causal mission logic may later consume separately governed point,
region, or directed-link monitors; it must not execute the global grid in the
simulation loop.

v0.2 retains sampled, center-of-cell, clear-line-of-sight sensor coverage and
adds optional bracketed acquisition/loss refinement. It is not
exact continuous footprint area, mission tasking, communications coverage, or
operational availability.

## Supported v0.2 Envelope

The frozen envelope supports:

- Earth only;
- one source spacecraft and one fixed, body-mounted sensor per analysis;
- OEL ECI spacecraft states with an absolute epoch and governed frame path;
- achieved ONP attitude, explicit replay attitude, or an explicit OGP
  analytic ideal attitude law;
- a circular axisymmetric hard-cone field of view;
- optional maximum slant range;
- WGS84 ellipsoid physical line of sight;
- a complete global HEALPix NESTED grid on the WGS84 authalic sphere;
- center-of-cell inclusion at explicit analysis epochs;
- sparse per-cell sampled-coverage intervals;
- optional provider-bisected acquisition/loss boundaries with explicit
  tolerance, iteration limit, provider identity, and disposition;
- global, declared-region-mask, and point-cell queries;
- per-cell sampled dwell and complete revisit-gap evidence;
- equal-area global and region summaries;
- deterministic chunking and artifact ordering; and
- mandatory cadence and grid-resolution sensitivity evidence.

No core analysis-resolution default exists. Every analysis must declare its
HEALPix order and time epochs. The first acceptance fixture uses order 6 and
must be repeated at order 7. Order 5 is suitable for smoke tests; orders 5
through 8 are the initially supported execution range. Decision-grade v0.2
baselines use orders 5 through 7 so that the required next-order refinement
remains inside the supported range; order 8 is a refinement or diagnostic
result rather than a terminal-resolution claim.

## Normalized Study Record v0.2

Scenario YAML, Python, and future agent adapters normalize into this closed
logical record before execution. This is the contract shape, not authorization
for arbitrary provider imports:

```yaml
contract_version: oel.global-earth-coverage-analysis.v0.2
analysis_id: unique-coverage-study
epoch:
  time_system: utc
  jd_utc: 2460000.5
evaluation:
  mode: postprocess
  times_s: [0.0, 60.0, 120.0]
  transition_refinement:
    enabled: true
    time_tolerance_s: 0.1
    max_iterations: 50
    max_evaluator_calls: 5000000
source_asset:
  asset_id: spacecraft_a
  state_provider:
    kind: scenario_object
    object_id: spacecraft_a
  attitude_source:
    kind: achieved               # achieved | replay | analytic_ideal
    provider_id: spacecraft_a.attitude_truth
sensor:
  sensor_id: spacecraft_a.imager
  parent_frame: body
  quat_body_from_sensor: [1.0, 0.0, 0.0, 0.0]
  pattern:
    kind: axisymmetric_hard_cone
    half_angle_deg: 20.0
    max_range_km: null
earth:
  physical_model: wgs84_ellipsoid_v1
grid:
  identity: healpix_nest_wgs84_authalic_v1
  order: 6
queries:
  region_masks:
    - region_id: example_region
      mask_version: 2026-08-19.v1
      provenance: analyst-declared-mask
      cell_indices: [100, 101, 102]
  points:
    - point_id: example_point
      longitude_deg: -104.526
      geodetic_latitude_deg: 38.803
```

The normalized record requires explicit `times_s` and `grid.order`;
convenience adapters may expand start/stop/cadence or prepare a versioned cell
mask before validation. `times_s` is strictly increasing, includes both horizon
endpoints, and contains at least two epochs. `mode` is only `postprocess` in
v0.2. Unknown fields fail validation. A transport adapter may not silently add
an attitude law, sensor constraint, environmental model, or grid default.

## Canonical Earth Surface

### Physical geometry

Physical visibility uses the WGS84 reference ellipsoid at zero ellipsoidal
height:

```text
semi-major axis a = 6378.137 km
flattening f       = 1 / 298.257223563
semi-minor axis b  = a (1 - f)
e^2                = f (2 - f)
```

Spacecraft position, ellipsoid intersection, slant range, and local surface
points use OEL's governed WGS84 geodesy and Earth-fixed/ECI frame path. Terrain,
geoid height, topography, atmosphere, and refraction do not modify the surface.

Public surface-target geometry helpers require the normalized ellipsoid level
to be within `1e-12` of one and the supplied unit normal to align with the
normalized ellipsoid gradient within `1e-10`. These checks run before Python or
native geometry dispatch.

### Equal-area indexing surface

Stable cell identity uses HEALPix NESTED indexing on the WGS84 authalic sphere.
Geodetic longitude is unchanged. WGS84 geodetic latitude `phi` maps to
authalic latitude `beta` using:

```text
e = sqrt(e^2)

q(phi) = (1 - e^2) [ sin(phi) / (1 - e^2 sin(phi)^2)
                     - (1 / (2e))
                       ln((1 - e sin(phi)) / (1 + e sin(phi))) ]

q_p    = q(pi / 2)
beta   = asin(q(phi) / q_p)
R_q    = a sqrt(q_p / 2)
```

The inverse mapping from a HEALPix center's authalic latitude to WGS84
geodetic latitude uses a deterministic bracketed solve on
`[-pi/2, pi/2]`. Pole values are handled exactly. The implementation records
its angular tolerance and iteration limit and fails rather than returning an
unconverged center.

Using the frozen WGS84 constants, the reference values are:

```text
authalic radius       = 6371.007180918474 km
ellipsoid surface area = 510065621.724088430 km^2
```

The cell surface areas are exactly equal under this mapping and sum to the
declared WGS84 ellipsoid surface area. Physical line-of-sight geometry still
uses the ellipsoid rather than the indexing sphere.

### HEALPix identity

For integer `order >= 0`:

```text
nside = 2^order
npix  = 12 nside^2 = 12 4^order
```

The stable cell identity tuple is:

```text
("healpix_nest_wgs84_authalic_v1", order, nested_pixel_index)
```

`nested_pixel_index` is the canonical zero-based HEALPix NESTED index in
`[0, npix)`. The representative point is the canonical HEALPix cell center,
inverse-mapped to WGS84 geodetic latitude at longitude unchanged and
ellipsoidal height zero.

Longitudes are normalized to `[-180, 180)` degrees, so positive 180 degrees is
represented as negative 180 degrees. Geodetic latitude is in `[-90, 90]`
degrees. HEALPix's canonical point-to-cell rule resolves cell-edge or vertex
ties; OEL must not create a second local tie-break rule.

The initially qualified order range and nominal values are:

| Order | Cells | Equal cell area (km^2) | sqrt(area) (km) |
|---:|---:|---:|---:|
| 5 | 12,288 | 41,509.246560 | 203.738181 |
| 6 | 49,152 | 10,377.311640 | 101.869091 |
| 7 | 196,608 | 2,594.327910 | 50.934545 |
| 8 | 786,432 | 648.581977 | 25.467273 |

The square-root value is a scale indicator, not a claim that HEALPix cells are
squares or have a constant diameter.

Although HEALPix identity is mathematically defined outside this table,
normalized v0.2 execution rejects orders outside 5 through 8. Supporting other
orders requires a later resource-envelope revision; it is not an automatic
consequence of backend support.

Cell identity and scientific results must not depend on a particular Python
package. An implementation may use a governed backend only after parity with
official HEALPix reference vectors is established. This freeze does not add or
authorize a runtime dependency.

## Tessellation Decision Record

The pre-implementation comparison selected HEALPix NESTED because v0.2 needs
global completeness, equal physical-area aggregation, hierarchy, deterministic
integer identity, and polar coverage more than it needs conventional GIS
polygon operations.

- HEALPix provides exactly equal-area, iso-latitude cells and a nested
  hierarchy with `12 nside^2` pixels.
- H3 provides a useful hierarchical hexagon/pentagon system, but its own
  documentation describes approximate geometric containment across levels and
  projection-dependent cell centers; its cells are not the frozen equal-area
  surface required here.
- S2 provides strong global spherical geometry and hierarchy, but its cells
  have varying area. Its official project also describes the Python API as
  unstable, which is undesirable for a first durable OEL artifact identity.

The references used for this decision are listed under References. A later GIS
export may map these stable cells into polygons without redefining the v0.2
coverage result.

## Sensor, Attitude, and Mounting

The source spacecraft uses OEL's scalar-first Hamilton achieved-attitude
quaternion `q_bn`; `C_bn` maps ECI vectors into body axes. The sensor mounting
is scalar-first `quat_body_from_sensor`; `C_bs` maps sensor vectors into body
axes. The sensor boresight is `+Z` in sensor coordinates:

```text
boresight_eci = C_bn^T C_bs [0, 0, 1]
```

The field of view is an axisymmetric hard cone with declared
`0 < half_angle_deg < 90`. An optional maximum slant range must be positive.
The angular boundary and maximum-range boundary are inclusive within their
declared numerical tolerances.

The full mounting quaternion is required. Achieved attitude is used when a
physical ONP attitude history exists. Commanded attitude alone is not accepted
as achieved coverage. OGP supplies no implied physical attitude; an OGP study
must declare an ideal analytic law or replay, and its result is labeled
`assumed_attitude` rather than achieved.

Missing, ambiguous, non-finite, or invalid attitude fails closed. Replay
interpolation, if used, is shortest-arc SLERP after quaternion-sign continuity
normalization. Every sample preserves attitude-source identity and whether it
is achieved, replayed, or idealized.

## Authoritative Cell-Coverage Predicate

At analysis epoch `t_i`, let `S_i` be spacecraft ECI position and let `P_ij` be
cell `j`'s WGS84 representative point transformed to ECI at the same epoch.
Let:

```text
d_ij       = P_ij - S_i
range_ij   = norm(d_ij)
look_ij    = d_ij / range_ij
off_axis   = acos(clamp(dot(boresight_eci_i, look_ij), -1, 1))
```

Cell `j` is covered at `t_i` if and only if all of these are true:

1. spacecraft and attitude state are valid at `t_i`;
2. the spacecraft is outside the WGS84 ellipsoid;
3. the open segment from `S_i` to `P_ij` does not intersect the ellipsoid
   before the representative surface point;
4. `off_axis <= half_angle`; and
5. an optional maximum range is absent or `range_ij <= max_range`.

This center predicate is the sole authoritative v0.2 coverage classification.
Polygon intersection, any-overlap, fractional overlap, and swept-footprint
rules must not be mixed into the result. A footprint boundary may be emitted
for visualization, but it is derivative and cannot override center results.

The complete Earth cell set is evaluated logically. Candidate-cell pruning,
vectorization, and chunking are implementation details and must produce the
same ordered covered-cell set as exhaustive evaluation within the frozen
tolerance.

## Time, Intervals, and Censoring

The study declares a strictly increasing sequence of analysis epochs that
includes both horizon endpoints. No cadence is inferred from propagation or
output settings.

v0.2 evaluates independent instantaneous footprints. For sampled metrics,
the cell state at `t_i` applies to the left-closed, right-open sample interval
`[t_i, t_(i+1))`. The final epoch retains a point snapshot but adds no duration.
There is no swept-footprint inference. A narrow or fast footprint can pass over
a cell between epochs and be missed. When transition refinement is enabled,
only a sampled boolean change bracket is bisected. The evaluator must use the
same coverage predicate and an arbitrary-epoch state/attitude provider. A
retained history may use declared cubic Hermite position/velocity interpolation
and shortest-arc attitude SLERP, but its evidence must say so; it is not an
exact-propagator claim. Convergence, iteration-limit, and censoring dispositions
are retained per interval.

Contiguous covered sample intervals are merged only when one ends exactly at
the next one's start. The merge tolerance is zero in sample-index space; no
short uncovered interruption is removed.

An interval touching the study start is `start_censored` because its true
acquisition may predate the horizon. An interval touching the study end is
`end_censored` because its true loss may follow the horizon. Dwell within the
declared horizon remains computable, but the censored interval is not used as
a complete outside-horizon event observation.

### Frozen numerical behavior

The normalized v0.2 configuration records these numerical tolerances:

- quaternion norm validation: `1e-10` absolute;
- authalic inverse-latitude solve: `1e-13 rad` absolute;
- angular boundary comparison: `1e-12 rad` absolute; and
- range boundary comparison: `1e-9 km` absolute.

An inclusive cone or range constraint passes when it is no more than its limit
plus the corresponding tolerance. Tangency with the WGS84 ellipsoid blocks
line of sight. Covered-cell identity, interval sample indices, dispositions,
and primary metrics are exact parity requirements across scalar reference,
vectorized, pruned, chunked, and parallel implementations.

## Frozen Metric Definitions

For cell `j`:

- `sampled_dwell_s` is the sum of covered sample-interval durations within the
  study horizon.
- A complete revisit gap is `next_interval.start - previous_interval.end` only
  when both bounding acquisition/coverage intervals occur inside the horizon.
- `max_complete_revisit_gap_s` is the maximum complete revisit gap. It is
  `null` with disposition `not_evaluated` when none exists.
- Prefix and suffix gaps are reported separately with boundary-censored
  dispositions; they are never substituted for a complete maximum revisit.
- `observed_acquisition_count` excludes a start-censored first interval because
  its acquisition was not observed inside the study.

At epoch `t_i`:

```text
instantaneous_covered_fraction_i = covered_cell_count_i / npix
```

Across the horizon:

- `time_weighted_mean_covered_fraction` weights each instantaneous fraction by
  its following sample-interval duration;
- `ever_covered_fraction` is the number of cells covered at least once divided
  by `npix`;
- `never_covered_fraction = 1 - ever_covered_fraction`; and
- covered area equals the corresponding cell count times the exact equal cell
  area.

Minimum/mean/maximum dwell or revisit summaries always state the included cell
population. Cells with no complete revisit gap are excluded from a complete-
revisit statistic and counted in its `not_evaluated_cell_count`.

### Regions and points

A region query is a versioned, sorted set of canonical cell identities at the
analysis order. Its provenance and semantic hash are required. Region metrics
apply the same definitions to that set; equal-area fraction is covered member
count divided by total member count. A region query cannot support a global
claim.

A point query contains WGS84 longitude, geodetic latitude, and zero
ellipsoidal height and maps to its containing canonical cell. Its result is
cell-level and resolution dependent; it does not prove subcell point
visibility. A later exact-point monitor is a distinct geometry product.

## Artifact and Review Contract

The stable logical products are:

- `coverage_analysis_manifest.json`: contract version, analysis/source/sensor
  identities, normalized configuration, horizon/epochs, state and attitude
  provenance, frame/Earth/authalic/HEALPix metadata, evaluator version,
  resource estimate, exclusions, input hashes, and artifact semantic hashes;
- `coverage_summary.json`: global metrics, dispositions, cell-population
  counts, cadence/order, sensitivity references, and claim limits;
- `coverage_samples`: time index, time, covered-cell count/fraction, area, and
  source state/attitude disposition;
- `coverage_cells`: cell identity, center coordinates, sampled dwell, interval
  count, observed acquisitions, complete revisit metrics, censoring, and
  dispositions; and
- `coverage_intervals.npz`: sparse interval arrays keyed by canonical cell.
  The sampled arrays remain stable; optional `refined_*` arrays retain cell,
  interval, continuous boundary, censoring, disposition, and reason evidence.

Phase 2 query products are:

- `coverage_query_manifest.json`: source analysis and interval semantic hash,
  order/grid identity, query semantic hash, query counts, artifact hashes, and
  completion disposition;
- `coverage_queries.json`: complete versioned region masks and their provenance
  and semantic hashes, regional summaries, normalized WGS84 point definitions,
  containing-cell identities, and point-cell summaries;
- `coverage_region_samples.csv`: per-region sampled covered-cell count,
  equal-area fraction, and physical covered area; and
- `coverage_point_samples.csv`: per-point containing-cell sampled state.

The Phase 2 query semantic hash binds the source interval semantic hash,
canonical query ordering, region-mask semantic hashes, normalized point
coordinates, and resolved point-cell identities. Individual artifact byte
hashes additionally protect the materialized JSON and CSV records.

The sparse interval artifact has canonical arrays:

- `cell_index`: ascending signed 64-bit NESTED indices;
- `interval_offset`: signed 64-bit offsets of length `len(cell_index) + 1`;
- `start_sample_index`: signed 64-bit inclusive starts; and
- `end_sample_index_exclusive`: signed 64-bit exclusive ends.

Arrays use little-endian representation and C order. The semantic artifact
digest covers the normalized manifest identity plus each array's name, dtype,
shape, and uncompressed bytes in the listed order. It does not rely on ZIP
container bytes or timestamps. The review store retains queryable identities,
summaries, dispositions, provenance, and the content-addressed artifact
reference rather than a dense `time x cell` matrix.

Partial or cancelled studies receive an incomplete disposition. Temporary
chunks are not complete evidence and must not be referenced by an ordinary
coverage summary.

## Determinism, Resources, and Sensitivity

Before execution, the analysis reports `npix`, epoch count, candidate dense
comparison count, estimated peak memory, output estimate, chunk plan, and
configured resource limits. It fails closed before materialization when the
request exceeds the authorized envelope.

Cell processing order is ascending NESTED index. Time order is analysis-epoch
order. Parallel worker count and chunk size may change performance but not
cell membership, interval boundaries, ordering, metrics, or semantic hashes.

Every decision-grade summary includes:

- the declared order and cadence;
- a cadence sensitivity comparison using a finer epoch sequence; and
- a spatial sensitivity comparison at the next HEALPix order.

The fixture contract uses order 6 as the baseline and order 7 as the spatial
refinement. Sensitivity evidence reports absolute and relative changes for
instantaneous extrema, time-weighted mean fraction, ever-covered fraction,
dwell summaries, and complete revisit summaries. The contract does not impose
a universal convergence threshold before empirical validation; the analyst
must retain and interpret the observed change.

## Validation and Failure Behavior

Configuration validation fails closed for missing epoch/frame provenance,
unsupported Earth or grid identity, undeclared order/epochs, invalid sensor
angles or ranges, duplicate identities, invalid source states, invalid or
ambiguous attitude, or an unsupported pattern.

Acceptance requires, at minimum:

- official HEALPix NESTED center/index reference-vector parity;
- WGS84 authalic forward/inverse round trips including poles and equator;
- total cell-area accounting equal to the frozen WGS84 surface area;
- stable identity across backends, chunks, and worker counts;
- known nadir, limb, off-axis, mounting, and attitude cases;
- exhaustive-versus-pruned covered-cell parity;
- antimeridian and polar fixtures;
- exact sampled-interval, no-gap-merge, dwell, revisit, and censoring cases;
- global versus region-mask aggregation equivalence;
- point-to-cell identity cases;
- order 6/order 7 and cadence-refinement evidence; and
- one independent matched-assumption comparison whose reference, inputs,
  tolerances, and discrepancies are retained.

Scientific tolerances and performance budgets are separate acceptance
records. Passing software tests alone does not establish exact continuous area
or operational sensing performance.

## Explicit Non-Claims and Deferred Work

v0.2 does not model or claim:

- exact footprint/cell overlap area or continuous swept coverage;
- an acquisition/loss event outside a sampled change bracket, or exact-provider
  timing when refinement used retained-history interpolation;
- terrain, geoid, buildings, clouds, atmosphere, weather, or refraction;
- rectangular, pushbroom, scanning, gimbaled, roll-dependent, or imported
  sensor patterns;
- illumination, sun-angle, incidence-angle, image quality, resolution, or
  payload performance;
- communications coverage without a separately declared Earth-terminal and
  Directed Link Analysis profile;
- tasking, slew feasibility, scheduling, data volume, storage, power, thermal,
  or packet delivery;
- multiple-spacecraft union, multiplicity, network, or constellation
  optimization;
- population, economic, or other non-area weighting;
- uncertainty probability, Monte Carlo availability, repeating-orbit, or
  steady-state inference beyond the simulated horizon; or
- operational sensing or mission assurance.

Agent-facing execution tools and public/Pro packaging are not frozen by this
scientific contract. They may be added only after the deterministic evaluator,
evidence products, and acceptance fixtures satisfy this boundary.

## References

- U.S. Geological Survey, Snyder, *Map Projections: A Working Manual*,
  Bulletin 1532: <https://pubs.usgs.gov/bul/1532/report.pdf>
- HEALPix introduction and mathematical properties:
  <https://healpix.sourceforge.io/doc/html/intro.htm>
- HEALPix geometric and algebraic properties:
  <https://healpix.sourceforge.io/doc/html/intro_Geometric_Algebraic_Propert.htm>
- H3 core-library overview and indexing documentation:
  <https://h3geo.org/docs/core-library/overview/>,
  <https://h3geo.org/docs/api/indexing/>
- H3 hierarchy and variable cell-area documentation:
  <https://h3geo.org/docs/>,
  <https://h3geo.org/docs/core-library/restable/>
- S2 cell hierarchy and area semantics:
  <https://s2geometry.io/devguide/s2cell_hierarchy>
- S2 project language/API status: <https://github.com/google/s2geometry>
