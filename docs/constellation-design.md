# Constellation And Ground-Network Design

OEL's public constellation-design workflow evaluates a small, explicit set of
Walker Delta, Walker Star, or circular shell candidates. It generates each
member state, propagates the members with ONP, evaluates ideal-nadir global
coverage and same-epoch ground links through the existing public analysis
owners, applies a transparent objective, and ranks every supplied candidate.

It is a bounded design trade, not a global constellation optimizer.

## Quick start

```bash
python -m sim.constellation_design validate \
  examples/constellation_design/public_walker_ground_network_trade.json

python -m sim.constellation_design solve \
  examples/constellation_design/public_walker_ground_network_trade.json \
  --output-dir outputs/public_walker_ground_network_trade

python -m sim.constellation_design replay \
  outputs/public_walker_ground_network_trade
```

The Python API is `sim.constellation_design`.

`propagation.numeric_backend` defaults to `"rust"` for existing ONP,
global-coverage and directed-link kernels. Explicit `"python"` selects the
reference implementation. Both selections bind normalized input, link hashes
and replay. Missing
required kernels fail before member propagation. Evaluation and ranking remain
serial Python workflows with the same public resource limits. Pro constellation
searches inherit the setting through `public_evaluation.propagation`, including
checkpoint identity, resume, promoted-design reevaluation and replay.

## Problem contract

`oel.constellation_design_problem.v1` declares:

- an absolute Julian UTC epoch, duration, and uniform sample cadence;
- ONP two-body or ONP J2 propagation and a bounded RK4 integration step;
- one ideal nadir conical coverage definition and required multiplicity;
- a catalog of fixed WGS84 ground sites and one transparent free-space link
  budget;
- coverage, network-availability, satellite-count, and ground-site terms in a
  declared linear score, plus optional feasibility thresholds; and
- one to eight explicit candidate designs.

Each candidate declares 2-24 satellites, a plane count that divides the
satellite count, phasing, circular altitude, inclination, initial RAAN and
phase, and 1-4 selected sites. Walker Delta uses a 360-degree RAAN span;
Walker Star uses 180 degrees. A `shell` candidate must declare its RAAN span.

For `T` satellites, `P` planes, and phasing `F`, OEL places `T/P` satellites
equally within each plane. Plane `p` receives the phase offset
`p * F * 360/T` degrees. These conventions follow the standard Walker
`i:T/P/F` construction described in NASA constellation-design references,
including [NASA/TM-2018-219992](https://ntrs.nasa.gov/api/citations/20180008672/downloads/20180008672.pdf).

The parser rejects unknown fields, non-finite values, duplicate identities,
unknown site references, inconsistent plane counts, and work above the public
sample, cell-comparison, or link-sample bounds before propagation begins.

## Evidence and replay

`solve` writes a new directory containing exactly:

```text
constellation_design_manifest.json
normalized_problem.json
constellation_design_evidence.json
```

The evidence includes generated initial states, sampled coverage fractions,
sampled union ground-link availability, per-link semantic hashes, feasibility,
every score component, deterministic rank, resource estimates, and claim
limits. If every candidate misses a declared service threshold, the workflow
retains the ranking but emits no recommended design. The manifest binds the
byte length and SHA-256 of each artifact.

`replay` requires the exact closed inventory, verifies every receipt, reparses
the normalized problem, reruns generation, propagation, coverage, link
analysis, scoring, and ranking, then compares the regenerated artifact bytes.
It does not merely trust the stored rank.

## Validation without paid tools

The public acceptance strategy is intentionally independent of STK and ODTK:

1. Contract tests cover strict schemas, bounds, identity normalization,
   fail-closed site references, output-directory atomicity, receipts, and
   tamper rejection.
2. Analytic geometry tests check Walker RAAN spacing, within-plane spacing,
   adjacent-plane phase, circular radius, and the distinct Delta, Star, and
   explicit-shell span conventions.
3. The propagation path reuses ONP and its existing conservation, convergence,
   force-model, and open-reference validation envelope. This workflow does not
   create a second propagator.
4. Coverage tests recompute the time-weighted service fraction from retained
   per-sample fractions. Existing coverage-aggregation tests independently
   compare member unions and multiplicities and reject mismatched grids,
   epochs, and service definitions.
5. Network tests independently recompute union availability from the retained
   Boolean sample series. Directed-link geometry and RF terms retain their
   existing analytic and programmatic acceptance evidence.
6. Ranking tests repeat identical problems, permute order-insensitive inputs,
   inspect every score component, and require byte-identical authoritative
   replay. No solver status is treated as proof of global optimality.
7. Optional VC-2 cross-checks should generate matching initial states and
   short propagated arcs with Orekit, GMAT, or Tudat. They strengthen named
   cases but are not required for routine offline replay.

Run the focused acceptance suite with:

```bash
python -m pytest \
  sim/tests/test_constellation_design.py \
  sim/tests/test_coverage_link_remaining_phases.py \
  sim/tests/test_global_coverage_phase3.py
```

## Local Edition And Private-Service Boundary

Public contains the inspectable scientific primitive: deterministic
Walker/shell generation, explicit small candidate evaluation, ONP propagation,
ideal-nadir coverage, same-epoch free-space ground links, transparent scoring,
bounded evidence, and authoritative replay.

The local v1 edition also exposes the bounded optimization/search modules under
legacy `pro_*` names. They do not provide global optimality, automatic
deployment or site placement, crosslinks and routed flow, station
capacity/contention, calibrated demand/cost/equipment models, operational
outage/uncertainty assurance, or design authority. Hosted service
infrastructure and the Manifold remain private. Public evidence does not imply
operational service qualification.

## Claim limits

- The top-ranked result is conditional on the supplied candidates, weights,
  thresholds, cadence, and models.
- Orbits are circular at initialization; deployment, maintenance,
  stationkeeping, replenishment, and steady-state repeat cycles are not
  modeled.
- Sensor attitude is ideal analytic nadir, not achieved spacecraft attitude.
- Ground links omit weather, terrain, interference, polarization, protocols,
  hardware reliability, and schedule contention.
- Per-link delivered-data estimates are not routed or scheduled capacity.
- The workflow is engineering analysis support, not operational qualification
  or authorization.
