# Conjunction Assessment

OEL v1.0 provides a bounded public conjunction-assessment foundation. It can
inspect and canonically round-trip a CCSDS CDM 1.0 KVN message, refine one
two-object time of closest approach (TCA), construct encounter-plane geometry,
project declared covariances, integrate one transparent educational 2D
collision probability, and evaluate simple targeter-backed impulsive avoidance
candidates through full-window repropagation and explicit secondary rescreening.

This is an inspectable research and education workflow. It is not an
operational catalog-screening service, a maneuver recommendation, or a claim
that OEL reproduces an originating agency's probability or disposition process.

`collision_probability_2d(..., numeric_backend="rust")` selects a stateless
native conditional-Gaussian integrand using SciPy's low-level callback ABI.
SciPy still owns adaptive quadrature, both tolerance passes, convergence evidence
and fail-closed acceptance. Rust is the default; native probability evidence
records its backend. `assess_histories` and `assess_conjunction` accept an
independent `probability_numeric_backend="rust"` selector, forwarding through
baseline, avoidance and secondary assessments. Their existing `numeric_backend`
continues to select geometry only. `assess_cdm_message(..., numeric_backend="rust")`
selects native Pc. Missing callback support fails explicit selection.

## Quick Start

Run the bundled synthetic crossing and retain the JSON evidence:

```bash
python -m sim.conjunction assess \
  examples/conjunction/synthetic_crossing.json \
  --output outputs/conjunction/synthetic_crossing_evidence.json
```

Inspect or round-trip one CDM without executing a scenario:

```bash
python -m sim.conjunction inspect-cdm path/to/message.cdm
python -m sim.conjunction roundtrip-cdm path/to/message.cdm \
  outputs/interchange/canonical.cdm
```

Recompute instantaneous CDM geometry and educational Pc after supplying the
two hard-body radii:

```bash
python -m sim.conjunction assess-cdm path/to/message.cdm \
  --primary-radius-m 5 --secondary-radius-m 3
```

## Bounded CDM Profile

The `oel.ccsds-cdm-kvn.v0.1` profile supports UTF-8 Keyword Value Notation,
CDM version `1.0`, one relative-metadata section, exactly `OBJECT1` followed by
`OBJECT2`, Cartesian state at TCA, and one complete 6x6 RTN covariance per
object. It validates required keywords, dates, finite numeric values, declared
units, covariance ordering, symmetry, and positive-semidefinite covariance.

The analysis-ready subset requires `EME2000` or `GCRF` object states. XML/NDM
containers, more than two objects, and extended drag/SRP/thrust covariance
dimensions are rejected rather than discarded. Optional standard metadata and
user-defined fields in the bounded profile survive semantic round-trip; exact
comments, whitespace, numeric spelling, and keyword layout are not formatting
preservation claims.

The field, ordering, and unit contract is traced to the
[CCSDS 508.0-B-1 Conjunction Data Message Blue Book](https://ccsds.org/Pubs/508x0b1e2c2.pdf),
including its published corrigenda.

CDM inspection reports the difference between `MISS_DISTANCE` and the norm of
the reported RTN relative position. `assess-cdm` also recomputes miss distance
and relative speed from the two object states, rotates each RTN covariance into
ECI, projects the sum into the encounter plane, and reports differences from
the message values. It does not refine the reported TCA because a CDM does not
contain the surrounding ephemeris histories needed for that calculation.

## TCA And Encounter Geometry

The general assessment problem propagates primary, secondary, and explicitly
declared screening objects through ONP. TCA refinement uses position and
velocity samples from those histories with piecewise cubic Hermite
interpolation. OEL forms the fifth-degree derivative of squared separation in
each common history interval, evaluates every real stationary root in that
interval, and retains the endpoints. This exhaustive polynomial refinement
makes repeated local minima and a boundary minimum visible rather than relying
on a single unimodal optimizer call.

The encounter frame uses:

- `z` along relative velocity;
- `x` along the miss vector projected normal to relative velocity; and
- `y = z cross x`.

The evidence retains the basis, plane coordinates, along-velocity component,
orthonormality error, winning interval, boundary disposition, and refinement
resource counts. It also retains `relative_position_dot_velocity_km2_s` and
relative range rate. A minimum at either search-window boundary is an incomplete
TCA search: encounter-plane covariance and Pc are withheld until the window
brackets an interior closest approach.

## Covariance And Educational Pc

Each JSON problem supplies a symmetric 6x6 ECI Cartesian covariance in mixed
km and km/s units, explicitly declared to apply at TCA. OEL sums the two
position covariance blocks and projects the result onto the plane normal to
relative velocity. It rejects non-finite, asymmetric, non-positive-semidefinite,
or singular projected covariance.

The `foster_2d_gaussian_disk_conditional_adaptive` method integrates the
bivariate Gaussian over the circular combined hard-body region after reducing
it to one standardized marginal coordinate and a conditional-normal interval.
Standardization keeps concentrated probability mass visible instead of relying
on fixed physical-space nodes. The evidence includes fine/coarse results,
adaptive error evidence, truncation bounds, and the acceptance tolerance; the
calculation fails closed when convergence is not demonstrated. Its assumptions
are Gaussian independent errors, linear relative motion through the encounter,
plane projection, and a circular combined body.

OEL does not propagate or estimate covariance in this workflow. A candidate
whose TCA moves still uses the declared TCA covariance; that limitation is
retained in every packet.

## Avoidance Candidates

An avoidance candidate declares one ECI or RIC impulsive-burn component, burn
time, one Cartesian terminal component and offset at the baseline TCA, solver
tolerance, perturbation, and delta-v limit. OEL builds an
`oel.trajectory_targeting_problem.v1` single-shooting problem and retains its
complete convergence and authoritative-repropagation evidence.

A converged targeter result is not a risk acceptance. OEL then:

1. materializes the solved impulse;
2. creates a fresh full-window primary history through ONP;
3. checks its state against the targeter's independently repropagated state at
   the original terminal epoch;
4. recomputes TCA, miss geometry, covariance projection, and Pc against the
   primary secondary; and
5. repeats the full assessment against every explicitly supplied screening
   object.

Candidates that are invalid, fail to converge, or exceed the declared limit
remain in the evidence with a failure disposition. A successfully repropagated
candidate is marked `assessment_completed`, while `risk_disposition` remains
`not_evaluated_no_acceptance_criteria`. This is a local equality target, not
avoidance optimization; increasing one miss distance is not proof that a
candidate is safe or preferable.

## Evidence Contract

`oel.conjunction_assessment_evidence.v1` contains:

- normalized problem SHA-256 identity;
- baseline refined TCA, states, miss distance, relative speed, and encounter
  frame;
- projected covariance, principal sigmas, Pc method, assumptions, quadrature
  settings, and convergence estimate;
- targeter convergence/Jacobian/resource evidence for each candidate;
- authoritative full-history continuity, primary rescreen, and all declared
  secondary rescreens; and
- explicit scope limits and non-claims.

## Validation Envelope

The retained focused tests cover:

| Claim slice | Free/open validation anchor |
| --- | --- |
| CDM syntax and semantics | CCSDS 508.0-B-1 field/order/unit contract, semantic round-trip, malformed and unsupported-profile fixtures |
| TCA refinement | Closed-form rectilinear encounter and independent dense sampling |
| Encounter frame | Orthonormality, orientation, and projected-miss identities |
| Covariance | PSD/symmetry failures, RTN/ECI rotation identities, and projection invariants |
| 2D Pc | Isotropic zero-mean closed form and independent SciPy adaptive integration for an off-center anisotropic case |
| Avoidance | Known synthetic crossing, targeter convergence, independent full-window repropagation continuity, primary recomputation, and declared-secondary rescreen |
| Failure handling | Wrong units, missing/extended covariance, non-PSD covariance, invalid timing, and non-convergent candidate dispositions |

Run it with:

```bash
python -m pytest -q \
  sim/tests/test_ccsds_cdm.py \
  sim/tests/test_conjunction_analysis.py
```

No STK, ODTK, or paid service is required for this validation envelope. Paid
comparison remains appropriate only for a later claim that lacks a
reproducible analytic, published, numerical-integration, Monte Carlo, or open
implementation reference.

## Local Edition And Private-Service Boundary

Public core includes the bounded two-object CDM, TCA, encounter geometry,
educational Pc, small explicit rescreen, targeter-backed candidate, and JSON
evidence surfaces described here.

The local edition does not provide catalog-scale monitoring, event association
and updates, covariance calibration, nonlinear Pc qualification, or operational
stores and maneuver authority. Hosted service infrastructure and the Manifold
remain private. Any result from this bounded workflow remains engineering
evidence, not an operational disposition.
