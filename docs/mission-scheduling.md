# Bounded Multi-Asset Mission Scheduling

OEL can select and independently replay a small exact schedule across multiple
spacecraft and shared ground stations. This workflow connects proven
observation and link opportunities to inspectable resource and delivered-data
evidence. It does not create opportunities or execute spacecraft commands.

## Quickstart

```bash
python -m sim.mission_scheduling solve \
  examples/mission_scheduling/public_two_asset_collection_problem.json \
  --output-dir outputs/public_two_asset_collection
python -m sim.mission_scheduling replay \
  outputs/public_two_asset_collection
```

The example contains two observations, two ground stations, a deliberately
contended GS-1 downlink, storage and energy limits, duty-cycle limits, and
direct slew-plus-settling bounds. The optimum delivers both observations while
rejecting the contended downlink in favor of SAT-B's later GS-2 opportunity.

Python callers may select `solve_mission_schedule(problem, numeric_backend="rust")`
for native exact subset evaluation. Python retains
the normalized problem, winning schedule details, rejection reasons, hashes,
and independent replay. Native score comparisons retain the Python solver's
compensated-sum and deterministic tie semantics. The default is Rust.

## Build Directly From OEL Evidence

The source adapter removes manual window transcription. A source plan names
completed OEL collection-evidence JSON files and directed-link artifact
directories, plus their expected spacecraft and station identities. Build,
solve, retain the sources, and replay the complete chain with:

```bash
python -m sim.mission_scheduling build-solve source_plan.json \
  --output-dir outputs/source_built_schedule
python -m sim.mission_scheduling replay-sources \
  outputs/source_built_schedule
```

For a self-contained example that generates two real optical-collection
products and three real directed-link products before scheduling them:

```bash
python examples/python/mission_scheduling_source_chain.py \
  --output-root outputs/public_mission_source_chain
```

The source-built output retains byte-identical input products, a portable
normalized source plan, the source manifest, and the nested scheduling packet.
Replay works from those retained copies; the original source locations are not
needed.

When the question also requires battery and solar-array feasibility for one
scheduled asset, use the [source-to-schedule-to-power study](study-lifecycle.md#start-from-collection-and-link-evidence).
It retains the verified source products, schedule, power packet, and study
receipt together and requires a separately supplied completed-run orbit history.
For a single asset, the stricter [orbit-bound route](study-lifecycle.md#require-one-shared-orbit-for-collection-link-and-power)
generates collection and link products from the same retained ECI history used
by power. Set `orbit_history_semantic_sha256` in the source plan; source
conversion and replay then reject missing or mismatched product citations.

## Building A Problem

Use `oel.mission_scheduling_problem.v1`. Declare one scheduling horizon,
per-asset resource constraints, at most 18 assets, and at most 18
opportunities. Opportunity and asset IDs must be unique. Objects are strict:
unknown fields, non-finite values or aggregates, oversized text, and inventory
overflow fail validation. Every opportunity must stay within the horizon and
must carry a lowercase SHA-256 identifying its source product.

Observations require positive `data_volume_bytes`. Downlinks require a named
`station_id`, positive `downlink_capacity_bytes`, and zero objective value.
When an asset declares `maximum_slew_rate_rad_s`, all of its opportunities
must provide normalized `pointing_unit_eci` vectors.

Keep source extraction outside the optimizer: validate the originating
coverage, collection, or directed-link product first, preserve its time bounds
and semantic identity, and only then create scheduler opportunities.

`oel.mission_scheduling_source_plan.v1` performs that extraction for supported
OEL products. Collection inputs must disable their independent resource screen
so the scheduler owns one shared resource ledger. Link products must match the
declared UTC epoch, endpoints, and horizon. When a scheduling asset enables
slew constraints, each link source must provide a normalized absolute ECI
pointing vector because directed-link v0.1 does not retain one.

## Reading The Result

Start with `mission_schedule_manifest.json`. It identifies the normalized
problem, exact schedule, source products, status, objective, and receipts for
every other artifact. The schedule CSV contains the selected time-ordered
activities and post-activity resources. The delivery CSV reports produced,
delivered, and remaining bytes for every selected observation. Rejections name
the first constraint found when adding an unselected opportunity to the
optimum, or state that a feasible opportunity lost on the global objective.

An infeasible result means no subset met the declared minimum observation
count and all other constraints. It does not mean the underlying mission is
impossible outside the supplied opportunity set.

## Validation Without Paid Services

The exact 18-candidate envelope makes exhaustive validation practical. OEL's
tests use an independently implemented subset oracle, hand-calculated station
and delivery cases, property-style resource invariants, permutation checks,
infeasible fixtures, malformed-input checks, CLI round trips, artifact receipt
checks, and authoritative solver replay. The generated public export reruns
the same workflow. Paid STK/ODTK or scheduling services are neither required
nor used for this claim.

Source-adapter tests add a real propagation-to-collection-to-link-to-schedule
chain plus candidate/task parity, receipt tamper, epoch/horizon/endpoint,
bit-to-byte, input-order, source confinement, source retention, and replay
checks.

See [the frozen contract](contracts/mission-scheduling-contract.md) for exact
solver semantics and [the source-adapter contract](contracts/mission-scheduling-source-adapter-contract.md)
for product conversion. The local v1 edition includes these bounded workflows;
Hosted service infrastructure and the Manifold remain private.
