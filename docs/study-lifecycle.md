# Integrated Study Lifecycle

OEL v1.0 can retain a bounded completed analysis as one deterministic study
bundle. A bundle records the question, plan, completed domain evidence,
evidence-backed claims and non-claims, and a content receipt that detects stale
or altered material.

V1 supports completed public trajectory-targeting, conjunction-assessment,
mission-scheduling, constellation-design, orbit-lifetime, and spacecraft-power
evidence. It does not execute those analyses for you; produce evidence with
the documented domain workflow first.

## Run the canonical example

This example runs all three real public workflows and creates one lifecycle
bundle per capability:

```bash
.venv/bin/python examples/python/study_lifecycle_three_domains.py \
  --output-root outputs/study_lifecycle_three_domains
```

The command finishes with `status: verified`. Each study also reports
`replay_status: identity_verified` and its stable bundle semantic digest.

The [spacecraft-power canonical example](spacecraft-power.md) demonstrates the
fourth registered capability and deliberately keeps authoritative power replay
separate from lifecycle identity replay.

The [orbit-lifetime canonical workflow](orbit-lifetime.md) demonstrates the
fifth registered capability and similarly separates authoritative ONP
recomputation from lifecycle identity replay.

The [constellation-design workflow](constellation-design.md) is the sixth
registered capability and keeps its generation, propagation, coverage, link,
scoring, and ranking replay separate from lifecycle identity replay.

## Build a schedule-coupled power study

The public schedule-to-power workflow retains the complete scheduling and power
packets, runs each domain's authoritative replay, and builds a two-step study
bundle. The power step depends on the schedule step; verification requires the
power evidence to cite the retained schedule's semantic SHA-256. Start with a
completed mission schedule and a power history exported from a completed run:

```bash
oel power export-review-history outputs/my_run --object-id SAT-A \
  --output outputs/my_power_history.json
oel study build-schedule-power \
  --schedule-dir outputs/my_schedule \
  --problem my_power_problem.json \
  --history outputs/my_power_history.json \
  --schedule-epoch-jd-utc 2461041.5 \
  --observation-load-w 180 --downlink-load-w 120 \
  --output-dir outputs/my_schedule_power_study
oel study inspect-schedule-power outputs/my_schedule_power_study
```

The UTC epoch is an explicit analyst assertion because the mission-scheduling
problem uses elapsed seconds and has no absolute epoch. It must match the power
problem and retained orbit history. The schedule horizon must lie within the
power horizon, and the selected schedule must contain activities for the power
asset. The new output directory contains `schedule/`, `power/`, `study/`, and a
`workflow_manifest.json`; inspection replays both domains and verifies the
study and exact activity/load binding. An infeasible power result remains a
valid completed assessment, with an explicit infeasible claim and model limits.

This CLI uses existing completed schedule evidence. It does not execute a
scenario or turn a prototype `PLAN_VALID` result into execution authorization.

### Start from collection and link evidence

To retain the source products as part of the same reviewable chain, pass a
mission-scheduling source plan instead of a completed schedule. The source
adapter verifies the named collection and directed-link products, builds and
solves the schedule, and retains all five source packets in the example:

```bash
oel study build-source-schedule-power \
  --source-plan outputs/my_sources/source_plan.json \
  --problem my_power_problem.json \
  --history outputs/my_power_history.json \
  --observation-load-w 180 --downlink-load-w 120 \
  --output-dir outputs/my_source_power_study
oel study inspect-source-schedule-power outputs/my_source_power_study
```

Relative product paths in the source plan resolve from the plan file's parent;
use `--base-dir` if they resolve elsewhere. The source plan supplies the UTC
epoch, which must match the power problem and history. The output retains a
`source_schedule/` packet containing the original collection and link products
and a `schedule_power_study/` packet containing the selected schedule, power
analysis, and study bundle. Inspection replays each authoritative domain,
compares the exact retained schedule copies, and checks their content bindings.
It does not establish that the source opportunity geometry shares the orbit
used for power analysis; that consistency needs separate evidence.

### Require one shared orbit for collection, link, and power

The stricter single-asset route starts with a completed-run review store. It
exports a canonical ECI history, recording the source review/config SHA-256s
and binding the complete state samples to a semantic SHA-256. Export checks
that the effective config matches review-store provenance and that the source
files remain unchanged while they are read:

```bash
oel study export-orbit-history outputs/my_run --object-id SAT-A \
  --output-dir outputs/my_orbit_history
python -m sim.collection my_collection_problem.json \
  --orbit-history-dir outputs/my_orbit_history \
  --output outputs/my_sources/collection.json
oel study build-orbit-bound-link \
  --orbit-history-dir outputs/my_orbit_history --config my_link_config.json \
  --station-latitude-deg 0 --station-longitude-deg 79.53938163 \
  --station-height-km 0 --first-sample-index 90 --last-sample-index 120 \
  --output-dir outputs/my_sources/link
```

The collection problem's asset, UTC epoch, first ECI state, and duration must
match the retained history. Collection uses its samples and existing Hermite
refinement; its declared propagation settings are not executed in this mode.
The link command selects exact parent sample indices, derives fixed-site states
at those times, and retains the indices, site coordinates, input digest, and
parent history digest. A directional body-frame terminal requires retained
attitude evidence.

Set `orbit_history_semantic_sha256` in the one-asset scheduling source plan to
the digest in `orbit_history_manifest.json`, and point its collection and link
sources at the products above. Then run:

```bash
oel study build-orbit-bound-source-study \
  --orbit-history-dir outputs/my_orbit_history \
  --source-plan outputs/my_sources/source_plan.json \
  --problem my_power_problem.json \
  --observation-load-w 180 --downlink-load-w 120 \
  --output-dir outputs/my_orbit_bound_study
oel study inspect-orbit-bound-source-study outputs/my_orbit_bound_study
```

The source adapter rejects missing or mismatched orbit citations. Study
inspection then reruns collection from the retained history, recreates each
link artifact from the retained sample indices and fixed-site inputs, checks
that power consumed the complete history, and replays schedule, power, and
study evidence. Collection replay allows `1e-10` absolute or `1e-12` relative
floating-point differences in non-identity derived values; IDs, digests,
booleans, exact times and quantities, and array structure match exactly. The
exported history retains source review and
configuration digests; inspection verifies the retained history, while a later
source-run audit can compare those digests with the original run. Shared input
identity does not establish sensor, RF, or power-model accuracy.

## Author and validate records

Create request, plan, and claims JSON objects using the
[`oel.study_*` v1 schema](contracts/schemas/oel-study-lifecycle-v1.schema.json)
and the field semantics in the
[Study Lifecycle Contract](contracts/study-lifecycle-contract.md). An authored
plan may set `request_sha256` to `auto`; authored claims may similarly set
`plan_sha256` to `auto`. Validation resolves those placeholders to normalized
content digests.

```bash
.venv/bin/python -m sim.study validate-request request.json
.venv/bin/python -m sim.study validate-plan request.json plan.json
.venv/bin/python -m sim.study validate-claims request.json plan.json claims.json
```

The same commands are available through the unified CLI as `oel study ...`.

## Build a bundle

Bind exactly one completed JSON evidence file to each plan step:

```bash
.venv/bin/python -m sim.study build \
  request.json plan.json claims.json \
  --evidence trajectory-targeting=outputs/targeting/evidence.json \
  --output-dir outputs/studies/transfer-study
```

The destination must not already exist. A successful build prints a verified
summary and leaves the six lifecycle records plus retained evidence under the
new directory.

## Inspect, replay identity, and compare

```bash
.venv/bin/python -m sim.study inspect outputs/studies/transfer-study
.venv/bin/python -m sim.study replay outputs/studies/transfer-study
.venv/bin/python -m sim.study compare \
  outputs/studies/transfer-study \
  outputs/studies/transfer-study-variant
```

`inspect` and `replay` fail closed if a bound record, cited value, retained
evidence byte, schema, status, or artifact set no longer matches. `compare`
reports changed root records and changed evidence steps after verifying both
bundles.

Study replay is provenance replay: it verifies the retained lifecycle graph.
It does not rerun domain physics. Use the trajectory targeter's mandatory
repropagation evidence and the conjunction, scheduler, constellation-design,
lifetime, or power
replay surfaces when you need scientific recomputation.

## Claim discipline

Every claim must:

- map to one or more request acceptance criteria;
- cite a known plan step that covers every claimed acceptance criterion;
- resolve to an existing value in that retained evidence with a JSON Pointer;
- carry an author-declared validation-level label from the `VC-0` through
  `VC-4` vocabulary, subject to the cited capability's maximum; and
- coexist with at least one explicit non-claim.

A valid receipt proves content identity and internal lifecycle consistency. It
does not substantiate the selected validation-level label. Every capability in
the v1 registry is capped at `VC-1`, so the current verifier rejects `VC-2`
through `VC-4` claims even though those values remain in the versioned schema
vocabulary for future capability contracts. Within the allowed ceiling,
reviewers must assess whether the retained evidence actually supports the
authored level. A valid receipt also does not prove global optimality,
operational suitability, flight qualification, or authorization to act.

## Local Edition And Private-Service Boundary

The versioned records, strict local CLI/Python API, identity replay,
comparison, schema, and small canonical studies are public. The lifecycle does
not provide managed execution, team review/signoff, customer data governance,
or hosted collaboration. Local campaign workflows remain separate producers.
Hosted service infrastructure and the Manifold remain private.
