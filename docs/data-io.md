# Modular data IO

`oel io` imports, inspects, converts and exports local data through bounded,
versioned profiles. `python -m sim.data_io` is the source-checkout equivalent.
Discover the actual profiles and optional dependencies with:

```bash
oel io formats
```

These workflows use local input files and explicitly supplied dependencies or
kernels; they do not fetch external catalogs, observations or kernels.

The IO package is separate from simulation execution. A serializer cannot start
a simulation or perform an orbit fit. Derived products consume saved evidence:

```text
external file -> format adapter -> normalized data -> new validated scenario
completed review store -> selection -> serializer -> external file + receipt
completed review store -> explicit OGP fit -> fit/holdout evidence -> TLE/OMM
```

The last operation requires the Pro implementation and `tracking_od`
entitlement. Public format conversion, review exports and OGP scenario
materialization do not reproduce that private estimator.

## Profiles

All profiles are experimental engineering interchange, not comprehensive
standard implementation or operational qualification. Run `formats` for exact
read/write directions and required options.

| Format IDs | Supported content | Boundaries |
| --- | --- | --- |
| `oem-kvn`, `oem-xml` | OEM 2.0/3.0 Cartesian PV, optional acceleration in message-preserving conversion, complete Cartesian covariance | Select one continuous segment for trajectory use. PV conversion rejects acceleration loss. No implicit ephemeris replay. |
| `opm-kvn`, `opm-xml` | OPM 2.0/3.0 state, covariance, physical parameters and maneuver blocks | Imported maneuver blocks can become an explicit non-executing plan. Exporting a message never executes a maneuver. |
| `omm-kvn`, `omm-xml` | OMM 2.0/3.0 mean elements and supported ancillary fields | One object per message. Materialization preserves mean elements and selects OGP. |
| `gp-json`, `gp-csv` | Multi-record GP catalogs, including nine-digit catalog identifiers | The named `celestrak-gp` profile supplies only CelesTrak's documented Earth/TEME/UTC/SGP4 omissions. Other conventions must be explicit. |
| `tle` | Checksum-validated input and standard two-line output | Output rejects unrepresentable catalog IDs, epochs and field widths. An osculating state cannot be relabeled as a TLE. |
| `tdm-kvn`, `tdm-xml` | TDM 2.0 native AZEL/RADEC, range, Doppler, frequency and phase records within the advertised keyword profile | Retains native observables and correction metadata. Parsing does not reduce radiometric data or declare it ready for instantaneous geometric OD. The existing reduced-geometric OD profile remains separate. |
| `cdm-kvn`, `cdm-xml` | Two-object CDM 1.0, full 6x6 RTN covariance and supported metadata | No extended drag/SRP/thrust covariance. A generated assessment CDM retains educational Pc and declared-at-TCA covariance limits. |
| `aem-kvn`, `apm-kvn` | ADM 2.0 quaternion and body angular velocity | AEM is a history; APM requires one selected sample. Explicit reference-to-body direction, scalar-last CCSDS encoding, degree/second conversion. Other attitude representations are rejected. |
| `stk-ephemeris` | STK `.e` time-position-velocity, named axes and center, metre/kilometre input | Bounded PV profile, no covariance syntax. |
| `stk-attitude` | STK `.a` scalar-last quaternion and optional angular velocity | Explicit Earth reference axes and body-frame orientation. |
| `sp3` | SP3-c/d selected-satellite PV, clock values and source flags | Named input frame required. Missing velocities require explicit finite-difference derivation on a continuous arc. Output uses regular samples and an explicit three-character satellite ID. No invented clock or covariance estimates. |
| `sentinel-eof` | Sentinel XML OSV position/velocity input in metres | Input-only; explicit object and frame. |
| `rinex-observation` | RINEX 3.02–3.05 observation epochs, native values, loss-of-lock and signal indicators, receiver clock offsets and header records | No special-event records, receiver orbit solution, cycle-slip correction or raw-observable reduction. |
| `rinex-navigation` | Eight-line GPS/Galileo RINEX 3.02–3.05 navigation records | Preserves broadcast parameters and clocks; it does not implement a new broadcast propagator. |
| `spk`, `ck` | NAIF spacecraft SPK sampling/type-13 writing; rate-bearing CK sampling/type-3 writing | Explicit local kernels, center/target or spacecraft-clock/instrument IDs. No kernel discovery/network fetch. SPK does not carry covariance. |
| `trajectory-json`, `trajectory-csv`, `trajectory-parquet` | Named-frame PV with absolute epoch and elapsed SI time; full covariance retained in JSON/Parquet metadata or the CSV receipt | CSV needs its hash-bound receipt or explicit context and column mapping. |
| `normalized-json` | Preserved native tracking or RINEX values | Re-importable structured IO data, not an orbit estimate. |
| `maneuver-plan-json` | Explicit OPM maneuver epochs, frames, duration, integrated delta-v and mass change | No execution authority or inference of a finite burn acceleration profile. |
| `czml`, `geojson`, `kml` | Position visualization | Earth only. CZML requires GCRF or ITRF; GeoJSON uses ITRF-to-WGS84 positions. KML contains time-tagged ground-track points to avoid confusing ellipsoidal height with height above sea level. |

Base installation remains unchanged. Install optional IO backends only when
needed: `python -m pip install '.[io]'`. It supplies `sgp4` formatting,
`spiceypy` and `pyarrow`; every missing backend fails with a specific message.

## Inspect and convert

```bash
oel io inspect input.oem --format oem-kvn
oel io convert input.oem output.xml --from oem-kvn --to oem-xml
oel io convert catalog.json catalog.csv --from gp-json --to gp-csv
oel io convert input.omm input.tle --from omm-kvn --to tle
oel io convert external.e trajectory.json --from stk-ephemeris \
  --to trajectory-json --object-id SAT
oel io convert precise.sp3 trajectory.json --from sp3 \
  --to trajectory-json --object-id G01 --frame ITRF
oel io convert tracking.tdm tracking.json --from tdm-kvn --to normalized-json
oel io convert tracking.json tracking.xml --from normalized-json --to tdm-xml
oel io convert maneuvers.opm plan.json --from opm-kvn --to maneuver-plan-json
```

For advanced inputs, `--options adapter-options.json` accepts explicit adapter
options. Unknown options fail. There is no frame, epoch or unit guessing. For
example, a custom CSV column map requires the complete `oel.trajectory.v1`
context, with `schema`, object, center, frame, epoch, time system, elapsed-time
basis and component units, plus one unique input column for each of `time_s`,
`x_km`, `y_km`, `z_km`, `vx_km_s`, `vy_km_s`, `vz_km_s`. Export a trajectory
JSON or inspect its CSV receipt to obtain a concrete context.

Supported Earth transforms reuse `sim.frame_time`. GPS uses TAI minus 19 SI
seconds. UT1/ITRF conversion requires epoch-covering EOP. SPICE center/frame
translation and TDB conversion require the caller's local kernels; center
translation treats kernel center motion as deterministic and does not add
center uncertainty to covariance. TCB is outside this profile.
Every active kernel must be listed explicitly; unrecorded ambient kernels and
meta-kernel discovery are rejected without clearing another caller's kernel pool.
CK export rotates OEL's body-coordinate angular velocity into the segment
reference frame required by [NAIF CKW03](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/ckw03_c.html);
import rotates it back to the body frame.
Canonical trajectory epochs use OEL's microsecond-resolution time contract;
finer external epochs fail instead of being silently rounded. Native RINEX
observation timetags retain their format's seven fractional decimal places.

## Build a new study

```bash
oel io materialize catalog.json configs/catalog.yaml --format gp-json \
  --scenario-name catalog_study --run-output outputs/catalog_study \
  --start-epoch 2024-01-01T00:00:00 --duration-s 3600 --step-s 60
oel io materialize trajectory.json configs/continuation.yaml \
  --format trajectory-json --scenario-name continuation \
  --run-output outputs/continuation --duration-s 600 --step-s 10
```

Materialization writes and safely validates ordinary passive scenario YAML. It never
executes the scenario or imports configured plugins. Catalog materialization uses native mean elements and
OGP; Cartesian materialization seeds ONP from the first sample using Earth/EME2000
or supported body-centered SPICE:J2000 planetary axes. OPM mass is retained when supplied; otherwise
mass defaults to 100 kg as an explicit study assumption. Review is enabled and full logs,
plots and animations are disabled. Validate the saved YAML before execution
and follow the ordinary run lifecycle. Public single-scenario execution retains
deterministic serial object stepping. Source ephemerides and covariances are
not automatically replayed or assimilated into a new estimator.

## Shared object import/export

The shared interface separates input selection, scenario composition, and export
from completed evidence. It delegates format handling, promotion decisions,
frame conversion, scenario validation, and complete continuation to their existing
owners. Use preflight to see missing options, source compatibility, retained fields,
defaults, and omissions before creating a scenario:

```bash
oel io object describe-import initial.opm --format opm-kvn
oel io object import initial.opm configs/imported.yaml --format opm-kvn \
  --as target --scenario-name imported --run-output outputs/imported \
  --duration-s 600 --step-s 10
oel io object import initial.opm configs/updated.yaml --format opm-kvn \
  --into configs/rpo.yaml --as target --replace \
  --scenario-name updated --run-output outputs/updated
oel io object import initial.aem configs/with_attitude.yaml --format aem-kvn \
  --into configs/imported.yaml --as target \
  --scenario-name with_attitude --run-output outputs/with_attitude
oel io object describe-export outputs/imported --object-id target
oel io object export outputs/imported final.opm --format opm-kvn --object-id target --final
```

`--object-id` selects the source object; `--as` names the destination object.
Imports select one exact zero-based sample with `--sample-index` (default zero).
Existing scenarios require a matching absolute epoch and central body. No hidden
propagation aligns mismatched epochs. Frame transformations must be explicitly
requested, with epoch-covering EOP or local kernels when required. Replacing an
orbit retains the destination's role, controllers, sensing, knowledge, hardware,
attitude, other objects, and study timing. A declared physical field updates that
field; an absent mass preserves an existing mass and defaults to 100 kg only for
a new object. OPM drag/SRP fields and saved per-sample review mass are retained.
Attitude import attaches an initial quaternion/rate sample to an existing ONP
object; it does not replay the source history. A missing rate defaults to zero.

Imported model context is preserved when creating a new scenario. Composition
blocks differing declared source/destination assumptions; `--allow-model-change`
explicitly retains the destination model and records both contexts. OEL product
quality, freshness, provenance and compatibility gates remain enforced after
normalization or conversion. Adjacent hash-bound OEL receipts preserve those
decisions across external formats. These receipts are provenance records, not
cryptographic authorization or a defense against deliberately forged input.

Python callers use the same interface:

```python
from sim.data_io.objects import import_object, export_object
from sim.scenarios import ScenarioBuilder

selected = import_object("initial.opm", format="opm-kvn")
builder = ScenarioBuilder("rpo").duration(600, dt_s=10)
builder.import_object(selected, object_id="target")
builder.chaser_relative_ric([0.1, 0, 0, 0, 0, 0], mass_kg=100)
# Configure outputs, safely validate, then follow the normal trusted run workflow.
export_object("outputs/rpo", "final.opm", format="opm-kvn", object_id="target", final=True)
```

`compose_object` returns a derived mapping; `write_composed_scenario` writes a new
validated YAML and receipt. Input files and completed outputs are never modified.
Full runtime continuation is separate from an orbital seed. The same export router
supports `--format satellite-checkpoint`, `snapshot`, `state-product`, and
`power-history`, delegating their native contracts and compatibility checks.
Checkpoint import/export retains compatible final satellite runtime; snapshots
require explicit `--object-ids` and final/sample selection and retain atomic
multi-object context. Import those native products with `object import --format
satellite-checkpoint` or `snapshot` and a new duration, step, and output location.
They cannot be renamed or merged into an arbitrary destination scenario.
Power history remains a separate recorded-history product.

Completed public tracking-OD workflows (or their estimator directory) can be
exported directly through `object export` or `export-review`. They preserve the
single fitted epoch, estimated-state semantics and full uncalibrated covariance;
the IO operation never reruns OD. `object import --format tracking-od` can seed
an explicit prediction study only after the saved solver, holdout and quality
gates pass and all bound artifacts verify. Export that new run for a predicted
trajectory. Raw TDM/RINEX observations still require a separate OD action.

## Export completed evidence

```bash
oel io export-review outputs/study trajectory.oem --format oem-kvn --object-id SAT
oel io export-review outputs/study trajectory.e --format stk-ephemeris --object-id SAT
oel io export-review outputs/study trajectory.csv --format trajectory-csv --object-id SAT
oel io export-review outputs/study state.opm --format opm-kvn --object-id SAT --sample-index 10
oel io export-review outputs/study attitude.aem --format aem-kvn --object-id SAT
oel io export-review outputs/study attitude.apm --format apm-kvn --object-id SAT --sample-index 10
oel io export-review outputs/study trajectory.czml --format czml --object-id SAT --target-frame GCRF
oel io export-table outputs/study access.csv --format csv \
  --query 'SELECT * FROM ground_access ORDER BY object_id, time_s'
```

Use the run's actual schema before composing a table query; the example query
must match that schema. A bounded `SELECT` can export access, coverage, link,
schedule, residual or other available review tables as CSV/JSON/Parquet.
Truncated selections fail instead of producing apparently complete files.

Object selection is mandatory for ambiguous runs. `--start-s` and `--stop-s`
select a bounded saved trajectory interval. `--sample-index` refers to the
zero-based row in that selected interval; the receipt retains the actual
review sample indices. Frame metadata, rather than legacy column names,
determines the state frame. Rotating CR3BP is never exported as ECI. Planetary
review keeps its center and explicit J2000 axes; body/frame conversion requires
an explicitly compatible adapter and kernels.

JSON, CSV and Parquet are new artifacts. The original review database and its
schema are read-only. Every export gets an adjacent `.receipt.json` binding
the source evidence, selection/query, options and output digest. Existing
artifacts and input files are never replaced. Covariance retains full cross
terms and its source calibration disposition; a format cannot establish
estimator calibration.

Validated OEL state-estimate/completed-state envelopes also support:

```bash
oel io convert state-product.json state.opm --from oel-state-product \
  --to opm-kvn --sample-index 0
```

For a predicted OEM, materialize the accepted state through the existing
handoff workflow, complete its ONP run, and export that run's review store.
This makes prediction an explicit action rather than hidden export work.

Saved conjunction assessments and reduced-geometric measurement datasets have
separate product exporters:

```bash
oel io export-assessment assessment.json problem.json assessment.cdm \
  --initial-epoch 2024-01-01T00:00:00 --format cdm-kvn
oel io export-measurements normalized_tracking_dataset.json measurements.tdm \
  --station-id SITE --object-id SAT
oel io export-maneuvers outputs/study historical-maneuvers.opm --object-id SAT
```

CDM export verifies the saved assessment's canonical problem digest and uses
its already-computed TCA states. The absolute initial epoch must be supplied
because the bounded assessment problem does not carry one. Historical
maneuver export reads actual ideal impulsive-command review rows. It does not
convert them into an approved future flight plan.

## Review-store mean-element fit (Pro)

```bash
oel io fit-review outputs/onp derived/catalog-fit --object-id SAT \
  --template prior.omm --template-format omm-kvn --fit-until-s 840 \
  --max-position-rms-m 1000 --max-velocity-rms-m-s 2 \
  --formats omm-kvn tle gp-json
```

The supplied mean-element prior is an optimizer seed, not an assumed answer.
The operation selects PV from the completed review store, explicitly converts
it to TEME, and invokes OEL's existing OGP-SGP4/SDP4 estimator. At least six fit
samples and three untouched holdout samples are required. The user supplies
position and velocity acceptance limits for this application.

`fit-evidence.json` records the review identity, exact sample selection,
template, fit partition, solver/gates, native-model residuals and separately
evaluated encoded-product residuals. Every encoded catalog is parsed after
formatting and propagated again through OGP to evaluate serialization effects,
including TLE quantization, on both partitions.
Rejected fits retain their evidence and emit no accepted catalog files.
Successful products retain independent receipts. This is approximation of a
selected simulated arc, not measured orbit accuracy, calibrated covariance,
catalog publication or operational flight authorization.

## Prescribed trajectory replay

`TrajectorySampler` provides bounded piecewise cubic Hermite interpolation
using supplied PV. Callers must supply `max_gap_s`; extrapolation and bridging
declared discontinuities fail. Exact samples are preserved. Interpolation does
not claim accuracy between samples without independent evidence.

`sim.dynamics.ephemeris_replay.PrescribedTrajectoryDynamics` implements the
standard `DynamicsModel.step` interface for explicit Python engine composition.
It requires Earth/EME2000, a declared simulation epoch, matching initial state
and passive commands. It rejects thrust, torque and resource commands. This
profile is a programmatic engine interface; the scenario YAML materializer
continues to produce ordinary ONP/OGP configurations.

## Ownership and validation

`sim.data_io.contracts` owns normalized PV/attitude values. Format owners are
`catalog`, `navigation`, `tracking`, `attitude`, `stk`, `precise`, `rinex`,
`spice`, `tabular` and `visualization`. `review` owns saved-evidence selection;
`products`, `tables` and the private `fit` operation derive new outputs. `api`,
`registry` and `cli` orchestrate these owners. Existing CCSDS and frame/time
owners remain authoritative; the package does not add an alternative orbital
propagator or estimator.

`objects.contracts`, `objects.importers`, `objects.composition`,
`objects.exports`, `objects.od`, `objects.continuation` and `objects.discovery`
own the shared object interface. `promotion`, `properties` and `annotations`
retain promotion rules, physical fields and hash-bound provenance; `object_cli`
only routes calls. `ScenarioBuilder.import_object` delegates composition without
duplicating it. Complete checkpoints/snapshots remain owned by `sim.interchange`.

Focused tests cover sample/metadata/covariance round trips, invalid inputs,
source mutation/identity, selection, holdout separation, passive replay,
optional dependencies, SQL restrictions, binary NAIF read-back and an actual
completed ONP run followed by a fitted catalog product. Run the applicable
public tests and private fit tests separately for the installed edition.

An independent, offline Orekit syntax-consumer check is available:

```bash
python -m tools.validate_data_io --orekit-root /path/to/pinned/orekit \
  --output-dir /new/path/to/acceptance
```

It records JAR/input/source hashes, independent parse results, record counts
and selected values. This is bounded format evidence, separate from full-field
semantic coverage, interpolation accuracy, physical-model validation, live STK
acceptance, sensor calibration and operational qualification. The standards
references are the [CCSDS navigation publications](https://ccsds.org/publications/allpubs/),
[CelesTrak GP profile](https://celestrak.org/NORAD/documentation/gp-data-formats.php),
[STK external files](https://help.agi.com/stk/Content/stk/importfiles.htm),
[IGS formats](https://igs.org/formats-and-standards/) and
[NAIF toolkit documentation](https://naif.jpl.nasa.gov/naif/documentation.html).
