# Bounded Multi-Asset Mission Scheduling Contract

Status: **implemented public v1 foundation**.

Problem schema: `oel.mission_scheduling_problem.v1`.

Evidence schema: `oel.mission_scheduling_evidence.v1`.

## Product Boundary

The public scheduler chooses an exact deterministic schedule from at most 18
caller-supplied opportunities across at most 18 declared assets. Problem,
asset, and opportunity objects reject unknown fields, non-finite values and
aggregates, oversized text, and inventory overflow. It does not propagate orbits, discover access,
repair windows, or dereference source products. Each opportunity names the
asset that can execute it and carries the SHA-256 of the coverage, collection,
or link product from which it was derived.

The v1 opportunity types are observation, downlink, and other. Observations
produce data and carry nonnegative mission value. Downlinks reserve one named
ground station, have zero objective value, and declare transfer capacity.
Other activities can consume time and energy but neither produce nor transfer
data.

## Constraints And Data Semantics

Activities on the same asset cannot overlap. An asset may declare a maximum
direct angular slew rate and settling time; when it does, every opportunity for
that asset must carry a unit ECI pointing vector. Each asset also declares a
horizon energy budget, storage capacity and initial fill, and maximum
observation duty cycle.

Downlinks using the same station cannot overlap even when they belong to
different spacecraft. Storage changes at activity completion. A downlink can
only remove data already onboard, so it cannot deliver a later observation.
Initial stored data is drained before newly generated observation data. With
`require_observation_delivery_by_horizon`, every selected observation must be
fully delivered by later selected downlinks. Delivered and undelivered bytes
are reported per observation.

The v1 energy model is a total horizon budget. It is not a time-varying power,
battery, thermal, packet, routing, or crosslink model.

## Optimization And Determinism

The solver exhaustively evaluates every subset within the bounded envelope.
It maximizes declared mission value. Equal-value schedules prefer fewer
activities and then lexicographically smaller opportunity-ID tuples. Asset and
opportunity input order do not affect normalized problem identity, schedule
identity, or the selected optimum.

The solver reports complete or infeasible status, evaluated and feasible
subset counts, selected and rejected opportunities, objective, per-asset
resource summaries, observation delivery, source hashes, normalized-problem
identity, and schedule identity.

## Evidence And Replay

`python -m sim.mission_scheduling solve` writes:

- `normalized_problem.json`;
- `mission_schedule_manifest.json` and `mission_schedule_summary.json`;
- `mission_schedule.csv` and `mission_schedule_rejections.csv`;
- `mission_resource_summary.csv`; and
- `mission_data_delivery.csv`.

The manifest content-binds all artifacts. `python -m sim.mission_scheduling
replay` reparses the normalized problem, reruns the exact solver, recomputes
resource and delivery ledgers, and requires the selected optimum, input and
schedule digests, summary, schedule, rejection, resource, and delivery
artifacts to match their deterministic authoritative forms. Output publication
uses an atomic staging-directory rename and refuses to replace any existing
destination. Replay does not authorize execution of a schedule.

## Validation And Claims

The acceptance suite compares the solver with an independently written
exhaustive oracle, checks hand-calculated schedules, exercises every resource
and precedence constraint, proves input-order determinism, and covers
infeasible and malformed inputs. Generated-public-export tests ensure the
workflow has no accidental private dependency. No STK, ODTK, paid scheduler,
or external optimization service is required for this bounded exact claim.

The public claim is exact optimality only within the declared opportunity set
and 18-candidate envelope. It is not a claim about opportunity correctness,
attitude feasibility beyond the direct slew bound, operational readiness, or
large-scale scheduling performance.

## Local Edition And Private-Service Boundary

Public includes the typed problem, bounded exact solver, shared-station
contention, per-asset resource checks, observation-to-downlink accounting,
content-bound evidence, authoritative replay, tests, and a synthetic example.

The bounded local workflow does not provide large mixed-integer or heuristic
optimization, rolling-horizon replanning, disruptions and recovery, batteries
and thermal state, crosslinks and routing, uncertainty, customer networks and
demand, operational data governance, dashboards, or review-ready service
evidence. Hosted service infrastructure and the Manifold remain private.
