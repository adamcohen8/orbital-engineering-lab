# OEL Product Handoff

OEL's interchange layer provides versioned product contracts, read-only
inspection, bounded State Estimate to ONP materialization, typed scenario
patch application, atomic multi-object continuation, and confirmed maneuver
event export. Completed review-store runs can export one exact state sample or
one exact shared multi-object sample for a new validated continuation study. It
lets analysts and agents determine what a product is,
where it came from, whether
its identity and source fingerprints are current, and whether its quality and
semantics permit automatic promotion.

Inspection never changes files. Materialization writes a normal scenario and a
handoff manifest, validates the scenario, and never executes it. Execution
remains a separate analyst-controlled action.

`compare-handoff` is the release-evidence bridge between those two steps. It
compares the producer product, generated scenario, manifest, and optional first
consumer review row. It writes a versioned parity packet and never executes a
scenario.

## Commands

Inspect a product or handoff manifest:

```bash
.venv/bin/python -m sim.handoff inspect <product-or-manifest.json>
```

Validate a product and print machine-readable diagnostics:

```bash
.venv/bin/python -m sim.handoff validate-product <product.json> --json
```

These inspection commands are read-only. By default they resolve relative source-artifact
paths from the product's directory and compare SHA-256 fingerprints. Use
`--no-verify-sources` only when the source files are deliberately unavailable;
schema, identity, and semantic checks still run, but integrity may remain
unverified.

Exit codes are:

- `0`: valid and promotable product, or a readable valid inspection target;
- `2`: malformed, unsupported, semantically incomplete, incompatible, or
  identity-invalid document;
- `3`: valid product evidence that is blocked from automatic promotion, such
  as `review_required`, `ambiguous`, `rejected`, `unknown`, or stale evidence.

## Compare Handoff Semantics

After materialization, create a compact semantic-parity packet:

```bash
.venv/bin/python -m sim.handoff compare-handoff \
  --product outputs/handoffs/source_product.json \
  --scenario outputs/handoffs/continuation.yaml \
  --output outputs/handoffs/continuation.comparison.json --json
```

The comparison verifies product validity and promotability, source product ID
and file hash, manifest identity, output digest, markings, adapter-specific
state/epoch/model semantics, and the materialization execution boundary. State,
native OGP mean-element, relative-state, covariance-study, and typed-patch
materializations use the same packet shape.

After a separately authorized execution, add `--run-output-dir <output-dir>`.
For absolute and relative state products, the comparison then checks the first
consumer review row against the promoted producer state and binds the review
database hash into a new comparison identity. A result of `equivalent` means
the checked handoff semantics match; it does not establish producer accuracy,
full-trajectory equivalence, or operational suitability.

## Export a Completed-run State

A completed single-run review store can emit one `oel.completed_run_state`
product. Select the object explicitly whenever
the run contains more than one eligible object:

```bash
.venv/bin/python -m sim.handoff export-state outputs/source_run \
  --object-id target \
  --sample final \
  --output outputs/handoffs/source_run_final_state.json \
  --json
```

The mutually exclusive selectors are `--sample final`, `--sample-index N`,
`--time-s T`, and `--event-id ID`. Time selection requires an exact recorded
`time_s`; event selection requires one exact event ID with a sample association.
An event's object is used when present, but an explicit conflicting object is
rejected.

The exporter reads through the SELECT-only review API and verifies that the
stored normalized config matches `config_sha256`. It selects the final sample
per object—not the global maximum row—requires the recorded state frame to be
canonical ECI, and derives the state epoch as
`initial_jd_utc + time_s / 86400`. The product binds the review database hash,
config hash, state-row hash, selector request, selected sample, source run, and
data markings into its semantic identity. Completed-run exports also bind the
selected `mass_kg`. When the source declares `dry_mass_kg`, the product derives
and binds remaining `fuel_mass_kg`; passive continuations therefore do not
silently reset a propulsive satellite to its launch mass.

Covariance is included only when `object_state_covariance` has one matching,
mathematically valid full 6x6 ECI matrix for the selected object and sample.
Ordinary runs leave that additive table empty; covariance-analysis runs can
populate it. Sigma-only summaries are never expanded into a diagonal matrix.

For a trusted relative-time fixture, `--epoch-jd-utc` supplies an explicit UTC
anchor. The exporter rejects a missing anchor and rejects an override that
conflicts with a configured source epoch; the choice is recorded in product
provenance.

## Export an Atomic Multi-object Snapshot

Use one exact shared final, sample, time, or event selection when a continuation
must preserve multiple objects together:

```bash
.venv/bin/python -m sim.handoff export-snapshot outputs/source_run \
  --object-id chief --object-id deputy --sample final \
  --output outputs/handoffs/source_snapshot.json --json

.venv/bin/python -m sim.handoff materialize-snapshot-onp \
  --snapshot-product outputs/handoffs/source_snapshot.json \
  --scenario-name passive_pair_branch \
  --output outputs/handoffs/passive_pair_branch.yaml \
  --run-output-dir outputs/passive_pair_branch \
  --duration-s 600 --dt-s 10 --trust-plugins --json
```

`oel.completed_run_snapshot` requires at least two canonical ECI states at the
same sample and epoch and binds available relative-pair rows from that sample.
The materializer creates one passive ONP object per state and never restores
controller, estimator, attitude, or mission-module memory.

## Export and Materialize a Satellite Checkpoint

Use the separately versioned `oel.satellite_checkpoint` product when the new
study must continue a complete GNC v2 satellite instead of only its orbit:

```bash
.venv/bin/python -m sim.handoff export-satellite-checkpoint outputs/source_run \
  --object-id chaser --context-object-id target \
  --output outputs/handoffs/chaser_checkpoint.json --json

.venv/bin/python -m sim.handoff materialize-satellite-checkpoint \
  --checkpoint-product outputs/handoffs/chaser_checkpoint.json \
  --scenario-name chaser_active_continuation \
  --output outputs/handoffs/chaser_active_continuation.yaml \
  --run-output-dir outputs/chaser_active_continuation \
  --duration-s 600 --dt-s 1 --trust-plugins --json
```

The product is final-sample only. It binds ECI position/velocity, attitude,
body rate, current wet/dry/fuel mass, the verified complete
`flight_software` configuration, opaque FSW state bytes, implementation and
configuration compatibility evidence, boot/load/invocation identity, onboard
clock position, pending input deliveries, actuator-command bus state, physical
actuator state, task releases, and sensor RNG state. State bytes are base64
encoded and independently SHA-256 bound.

Context objects may be included for relative navigation. They are continued as
passive truth-state references; their own onboard software is not restored.
Checkpoints that depend on external interactive publishers are rejected because
a scenario alone cannot recreate that external state. Materialization validates
but never executes the continuation.

## Export a Maneuver Detection

A confirmed review event can be promoted to a versioned detection product:

```bash
.venv/bin/python -m sim.handoff export-maneuver-detection outputs/detection_run \
  --observer-id chaser --target-id target \
  --output outputs/handoffs/maneuver_detection.json --json
```

The `oel.maneuver_detection` payload binds the event/sample, observer, target,
detector settings, summary evidence, source hashes, and non-claims. The Pro
Scale adapter can export measured event-centered observations, run dynamics
OD, and prepare a truth-free IHE input bundle and authoring skeleton. Weak fit
or holdout evidence remains `review_required`; no intent or hypothesis is
selected automatically.

## Emit a Scenario-capability Overlay

Use an overlay when an accepted scenario needs bounded, explicit access,
complete satellite flight software, knowledge, review, termination, or supported analysis
configuration without workflow-local YAML copying:

```bash
.venv/bin/python -m sim.handoff emit-overlay \
  --source-scenario outputs/base.yaml \
  --overlay overlay.yaml --overlay-id station_and_control \
  --rationale "Bind reviewed station and control context" \
  --output outputs/handoffs/station_and_control.json --json
```

The result is an `oel.scenario_patch` with
`patch_type: scenario_capability_overlay`. It is source-hash bound, rejects
unknown paths, and still requires separate materialization, validation, and
execution.

Use the normal ONP materializer on the exported product:

```bash
.venv/bin/python -m sim.handoff materialize-onp \
  --state-product outputs/handoffs/source_run_final_state.json \
  --scenario-name source_run_continuation \
  --output outputs/handoffs/source_run_continuation.yaml \
  --run-output-dir outputs/source_run_continuation \
  --duration-s 600 --dt-s 10 --trust-plugins --json
```

The generated scenario is a new study. Neither the exporter nor materializer
modifies the completed run or executes the continuation. The scenario metadata
and handoff manifest cite the original run, review-store hash, selector, sample
index, sample time, and event evidence when applicable.

## Materialize an ONP Scenario

An accepted, current, canonical ECI state product can be converted into a
passive ONP continuation scenario without copying state values by hand:

```bash
.venv/bin/python -m sim.handoff materialize-onp \
  --state-product outputs/agent_tasks/dynamics_od_smoke/state_estimate_product.json \
  --scenario-name dynamics_od_onp_continuation \
  --output outputs/handoffs/dynamics_od_onp.yaml \
  --run-output-dir outputs/handoffs/dynamics_od_onp_run \
  --duration-s 3600 \
  --dt-s 30 \
  --trust-plugins \
  --json
```

The command always performs safe validation. `--trust-plugins` explicitly
authorizes ordinary validation, including imports named by the generated
scenario; it still does not authorize execution. Omit it when inspecting an
untrusted product or source tree. Review the generated YAML and manifest before
running the separate command:

```bash
.venv/bin/python run_simulation.py \
  --config outputs/handoffs/dynamics_od_onp.yaml
```

The materializer preserves the Cartesian state, UTC Julian-date epoch, object
identity/specifications, compatible orbit force-model and environment
settings, quality disposition, source hashes, and data markings. It applies
only documented passive-study defaults: `fsw.passive`, disabled attitude
and rocket dynamics, standard review output, and disabled plots/animations.
Duration, output cadence, scenario name, scenario path, and run output path are
explicit caller choices.

If a source force-model record carries `orbit_substep_s` larger than the
explicitly requested consumer `dt_s`, the materializer bounds the generated
substep to `dt_s` so the scenario remains valid. The manifest records the
source value, output value, and cadence-bound reason as an explicit override;
the adjustment is never silent. Nonpositive or non-finite source substeps fail
compatibility validation.

Non-accepted or stale products, failed source fingerprints, incompatible frame
or attitude semantics, unsupported force-model mappings, and output conflicts
are blocked. When the input can be loaded, the command writes a manifest with
the failure and recommended next action but does not write or overwrite the
scenario. Replacing a different existing scenario requires `--overwrite`.

The default manifest path is `<scenario-stem>.handoff_manifest.json`. It records
the source product and file hash, adapter version, options/defaults, scenario
digest, validation results, markings, failures, next action, and
`execution_occurred: false`. Semantic IDs exclude timestamps and filesystem
locations, so relocating the same product and materialization options does not
change the product, scenario, or manifest identity.

## Select and Materialize a Scenario Patch

Mission-recovery reporting writes one typed patch per planner candidate and a
`mission_recovery_scenario_patches.json` index under the run's
`scenario_patches/` directory. Select one exact candidate; recommendation rows
are evidence and are never applied implicitly:

```bash
.venv/bin/python -m sim.handoff materialize-scenario-patch \
  --patch-index outputs/recovery/scenario_patches/mission_recovery_scenario_patches.json \
  --selection-id <candidate-id> \
  --source-scenario configs/recovery_source.yaml \
  --scenario-name selected_recovery \
  --output outputs/handoffs/selected_recovery.yaml \
  --run-output-dir outputs/handoffs/selected_recovery_run \
  --trust-plugins --json
```

An exact product can instead be supplied with `--patch-product`; do not pass
`--selection-id` in that form. The command verifies the source scenario's byte
hash and normalized semantic digest, applies ordered allowlisted operations,
writes a new scenario, and validates it. It never modifies the source scenario
or runs the materialized scenario.

`oel.scenario_patch` preserves the selection ID and rank, recommendation modes,
objective, constraints, producer evidence, source artifacts, and data
markings. Mission patches can append checked-in scheduled-burn modules and
extend duration. Satellite optimization patches replace a complete
`flight_software` stack; direct controller-pointer replacement remains limited
to rocket `base_guidance`. Patches may also apply ordered, allowlisted scenario
overrides. A stale source, unknown
selection, non-accepted product, failed controller rerun, or incomplete burn
duration blocks materialization and leaves an inspectable manifest.

The public planner and common patch contract are public-safe. Controller
optimization remains a Pro producer; it emits a standalone best-variant YAML
and one source-bound patch per benchmark case because no suite-wide variant can
be applied safely to an unrelated scenario.

## Product Envelope v1

The common envelope is closed at the top level. It binds:

- product kind and deterministic semantic identity;
- producer capability and OEL version;
- product-specific payload;
- original producer status, gates, warnings, and non-claims;
- integrity and age freshness as separate decisions;
- source artifacts, source product IDs, and named transformations;
- data scope and handling markings.

Product identity uses SHA-256 over canonical finite JSON. Key ordering,
presentation whitespace, creation timestamps, source paths, and freshness
evaluation timestamps do not affect identity. State values, frame, epoch,
covariance, quality evidence, source hashes, transformations, and data markings
do affect identity.

## State Estimate Product v1

The first semantic payload validator is `oel.state_estimate`. It requires:

- a six-component Cartesian position/velocity state;
- explicit `[x, y, z, vx, vy, vz]` ordering and km/km/s units;
- canonical `ECI` frame and a positive UTC Julian-date epoch;
- covariance bound to the same frame and epoch, or an explicit non-empty
  reason that covariance is absent;
- object specifications and producer model assumptions;
- estimator method, selected parameters, and a source-report hash bound into
  provenance.

TEME or another frame is not silently relabeled as ECI. A future named adapter
must perform and record any supported transformation.

## Completed Run State Product v1

`oel.completed_run_state` uses the same canonical ECI Cartesian state and
covariance conventions as State Estimate v1, but identifies its source as one
selected simulator truth row rather than an estimator result. Its closed
payload adds verified source-run metadata and an exact final, sample-index,
time, or event selection. Adapter v2 also binds current mass and, where dry
mass is known, remaining propellant. It does not claim to continue attitude, controller
memory, estimator memory, or mission-module state.

## Satellite Checkpoint Product v1

`oel.satellite_checkpoint` is the explicit active-continuation contract for a
complete GNC v2 satellite. Unlike an orbital state product, it carries truth
attitude/rates and hash-bound opaque FSW/runtime state. Its materializer retains
the source stack and hardware profile and installs the checkpoint at the typed
flight-software boundary. Stack implementation/configuration mismatches,
corrupt state bytes, inconsistent clocks, incompatible hardware identities, or
external publisher dependencies fail closed.
The game-pilot stack is also excluded because it requires the surrounding
interactive game session; game continuation remains a separate future seam.

## Relative State Estimate Product v1

The common validator also recognizes `oel.relative_state_estimate`. It requires:

- explicit, distinct chief and deputy object IDs;
- an accepted chief State Estimate Product ID in source-product provenance;
- a six-component rectangular RIC state at the chief product's positive UTC
  Julian-date epoch;
- radial outward, in-track along chief motion, and right-handed cross-track
  axes, with deputy-minus-chief sign convention;
- covariance in the same rectangular RIC basis and epoch, or an explicit
  reason it is absent; and
- relative dynamics metadata, estimator method, and a source-report hash bound
  into provenance.

The public facade validates and inspects this product kind but does not expose
the private producer or chief/deputy scenario adapter. Curvilinear RIC and
opposite-sign conventions are not silently relabeled as rectangular RIC.

## Scenario Patch Product v1

The common validator recognizes `oel.scenario_patch`. Its closed payload binds:

- the source scenario name, exact SHA-256, and normalized scenario digest;
- one patch type and ordered typed operations over allowlisted paths;
- an explicit candidate or case/variant selection with positive rank;
- objective, constraint, and producer evidence; and
- envelope quality, freshness, provenance, non-claims, and markings.

It is not a general YAML merge format. Unsupported paths and operations are
contract errors, while valid but `review_required` evidence is inspectable and
non-promotable.

## Examples and Schemas

The package includes:

- `sim/interchange/schemas/oel-product-envelope-v1.schema.json`
- `sim/interchange/schemas/oel-state-estimate-v1.schema.json`
- `sim/interchange/schemas/oel-relative-state-estimate-v1.schema.json`
- `sim/interchange/schemas/oel-scenario-patch-v1.schema.json`
- `sim/interchange/schemas/oel-ogp-mean-element-product-v1.schema.json`
- `sim/interchange/schemas/oel-completed-run-state-v1.schema.json`
- `sim/interchange/schemas/oel-completed-run-snapshot-v1.schema.json`
- `sim/interchange/schemas/oel-maneuver-detection-v1.schema.json`
- `sim/interchange/schemas/oel-handoff-manifest-v1.schema.json`
- `sim/interchange/schemas/oel-handoff-comparison-v1.schema.json`
- `sim/interchange/examples/state_estimate_accepted_current.json`
- `sim/interchange/examples/validation_fixture_matrix.json`

The examples are synthetic public-safe contract fixtures. They are not
operational orbit estimates and establish no propagation or OD accuracy claim.

## Python Facade

Use the stable facade for programmatic inspection:

```python
from sim.handoff import (
    compare_handoff,
    export_completed_run_snapshot,
    export_completed_run_state,
    inspect_path,
    load_interchange_document,
    materialize_scenario_patch,
    materialize_ogp,
    select_patch_product,
    materialize_snapshot_onp,
    validate_product,
)

summary = inspect_path("product.json")
document = load_interchange_document("product.json")
report = validate_product(document, source_path="product.json")

comparison = compare_handoff(
    "product.json",
    "continuation.yaml",
    output_path="continuation.comparison.json",
)

completed = export_completed_run_state(
    "outputs/source_run",
    output_path="outputs/handoffs/source_run_final_state.json",
    object_id="target",
    selector="final",
)

snapshot = export_completed_run_snapshot(
    "outputs/source_run",
    output_path="outputs/handoffs/source_run_snapshot.json",
    object_ids=["chief", "deputy"],
    selector="final",
)

# A fitted OGP mean-element product stays native TEME mean-element evidence;
# this writes and validates a passive OGP scenario without inventing TLE text.
ogp_result = materialize_ogp(
    "fitted_ogp_mean_element_product.json",
    scenario_name="ogp_continuation",
    scenario_path="ogp_continuation.yaml",
    output_dir="outputs/ogp_continuation",
    duration_s=3600.0,
    dt_s=60.0,
)

patch = select_patch_product("scenario_patches.json", "candidate-a")
result = materialize_scenario_patch(
    patch,
    "source.yaml",
    scenario_name="selected_candidate",
    scenario_path="selected_candidate.yaml",
    output_dir="outputs/selected_candidate",
)
```

Focused ownership remains under `sim.interchange`. The facade contains only
dispatch and compatibility exports and does not import producer-specific or
Pro-only adapters. The batch dynamics-OD emitter is a private producer adapter;
the common envelope, inspection, and ONP materialization contract remain
public-safe.

The Pro covariance-analysis adapter remains outside this public facade. In an
authorized private workspace, use
`python -m sim.estimation materialize-covariance-study`; see the Pro user guide
and `examples/workflows/pro_od_covariance_continuity/README.md`. This separation
keeps common State Estimate Products public-safe without exposing private
covariance campaign orchestration.

## Compatibility and Deprecation Window

The supported native numerical-propagation term is ONP. Current canonical
Scale interfaces use `onp_handoff`, `handoff-onp`, `run_onp_handoff`, ONP
packets, and ONP report labels. Historical `hpop_handoff`, `handoff-hpop`,
`run_hpop_handoff`, HPOP-named packet IDs, store tables, and internal fields
remain compatibility aliases because existing configs and persisted Scale
stores depend on them.

The v0.24 sentence was historical. In the current source checkout, no removal
release has been announced and the aliases remain compatibility exports without
an active removal warning. Removal requires
a complete config/artifact/store migration, an ONP-first replacement for every
consumer, explicit changelog notice, and at least two minor releases carrying
deprecation warnings. HPOP remains valid terminology for external MATLAB HPOP
reference and validation workflows; it is not the product name for OEL's
native propagator.
