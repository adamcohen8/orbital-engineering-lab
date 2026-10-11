# Public Core And Pro Boundary

Orbital Engineering Lab is an open-core project. The public repository is a
complete, inspectable simulation foundation; the private Pro repository adds
workflow acceleration, scale, and customer-specific engineering support.

This page is the authoritative product and repository boundary. When another
document disagrees with it, this page and the checked-in public-surface manifest
govern.

## Public Core

The public core includes:

- deterministic single-run orbit and attitude simulation;
- buildable Rust source for public numeric backends and the Trainer, including
  build manifests and redistributable dependency source and licenses;
- OGP-SGP4/SDP4 passive catalog-style propagation and configurable ONP
  numerical propagation;
- opt-in Earth-centered ONP spacecraft resources with one body thermal node,
  fixed solar panels, a bounded battery, and saved simulation-truth histories;
- a headless ground segment for geometric service availability, delayed
  tracking/resource packets, explicit outages, and two-body orbit estimates
  initialized from supplied priors;
- public controllers, sensors, estimators, actuators, and mission primitives;
- opt-in Earth albedo/infrared radiation pressure, Schwarzschild acceleration,
  solid Earth tides, and ocean tidal gravity with optional Rust kernels;
  ocean coefficients remain authorized external inputs;
- the Public FSW Authoring Kit for ADCS/RPO stack scaffolding, safe inspection,
  trusted lifecycle validation, component tests, and one deterministic serial
  smoke run;
- scenario YAML, CLI, Python API, review-store queries, and plotting;
- opt-in custom ONP force-model plugins for trusted local modules, with
  module-path and runtime-output validation;
- content-bound review slices for selected completed-run evidence;
- bounded CCSDS OEM 3.0 KVN inspection, import, completed-run export,
  Cartesian covariance interchange, semantic comparison, and public validation
  fixtures;
- bounded CCSDS OPM/OMM 3.0 KVN inspection and semantic round-trip, plus
  Earth/EME2000/UTC OPM state import while OMM remains a preserved mean-element
  product;
- bounded CCSDS TDM 2.0 KVN inspection and canonical round-trip for
  analyst-declared reduced-geometric UTC AZEL and one-way unambiguous range,
  plus one single-object batch fit with explicit holdout prediction evidence;
- bounded canonical UTC/TAI/TT/sampled-UT1 handling and
  EME2000/TEME/ITRF state and 6x6 covariance transforms with explicit EOP
  provenance;
- deterministic whole-Earth coverage, point/region/rich-footprint queries,
  free-space directed-link budgets, communications coverage, constellation
  aggregation, bounded tasking, cadence sensitivity, event refinement, and
  their ONP/OGP/review-history adapters;
- bounded exact multi-asset mission scheduling for up to 18 supplied
  opportunities, with per-asset slew/settling, energy, storage, and duty-cycle
  constraints, shared-station contention, delivered-data accounting, and
  authoritative replay, plus content-bound conversion from completed public
  optical-collection and directed-link products;
- bounded evaluation of up to eight explicit circular Walker/shell and
  ground-site candidates, with ONP propagation, ideal-nadir coverage,
  same-epoch free-space links, transparent score components, sampled evidence,
  and authoritative replay;
- bounded deterministic spacecraft-power feasibility for one retained ECI
  history and load timeline, with eclipse, array incidence, lumped battery
  limits, schedule binding, conservation evidence, and authoritative replay;
- bounded deterministic ONP orbit-decay/lifetime analysis for one supplied ECI
  state, with frozen atmosphere inputs, refined altitude thresholds, complete
  state/drag/orbit histories, identical-input model comparison, and
  authoritative replay;
- a bounded two-body Lambert transfer planner and mission-recovery estimates;
- deterministic event-driven coast/burn sequences, one transparent
  single-shooting impulsive targeter, JSON convergence/resource evidence, and
  mandatory authoritative solution repropagation;
- bounded CCSDS CDM 1.0 KVN inspection and semantic round-trip, deterministic
  two-object TCA/encounter geometry, one transparent educational 2D Pc, and
  targeter-backed impulsive candidates with full-window primary and explicit
  small-list secondary rescreening;
- bounded single-target optical collection opportunities with WGS84 hard-FOV
  footprints, illumination and pointing screens, transparent first-order
  resolution, refined transitions, and optional storage plus content-bound
  downlink screening;
- versioned local study request, plan, run, evidence, claims, and receipt
  records for completed public trajectory-targeting, conjunction-assessment,
  mission-scheduling, constellation-design, orbit-lifetime, and
  spacecraft-power evidence, including strict validation, content-bound
  inspection, identity replay, and semantic comparison;
- public examples, the RPO Trainer, and reproducible public validation evidence;
- the public-safe Hosted OEL client for capability discovery, explicit file
  grants, user-approved submission, durable status, and verified result import;
- the public Hosted execution package contract and offline local validator for
  exact file identity, typed public or Pro planning, visible bounds, a local
  recommendation for public-only plans, and explicit handoff to authoritative
  hosted preflight without distributing Pro executors;
- educational rocket/ascent primitives and public rocket GNC contracts.

The public core is intended for research, education, prototyping, and
inspectable engineering experiments. It is not flight-qualified or an
operational decision system.

## Pro Layer

Pro adds workflows whose value comes from repeatability, scale, search, or
review-ready packaging:

- private precision-model validation automation and customer-specific
  environmental studies; the four documented Earth precision forces are public;
- the experimental private `sim.trajectory_targeting` package for
  force-configurable impulsive coast/burn studies and reference comparisons;
- Monte Carlo, sensitivity, calibrated covariance analysis, and campaign
  orchestration;
- controller benchmarks, comparison reports, optimization, and gain tuning;
- OEL Scale catalog screening, refinement, operational stores, and synthetic
  data generation;
- intent-hypothesis evaluation and maneuver-investigation workflows;
- governed RF link budgets, atmospheric availability, RF-qualified coverage,
  bounded contact scheduling, communications campaigns, and declared
  ground-network or constellation trade evidence at managed scale;
- automatic constellation design-space generation, ground-site placement,
  crosslink/routed-capacity modeling, robust optimization, customer
  demand/cost/equipment models, and design dashboards;
- governed or bulk orbit determination against customer observations or
  precise products, broader raw-radiometric measurement reduction, calibrated
  covariance and predicted-accuracy qualification, and operational tracking
  workflows;
- curated validation automation and private release/customer evidence;
- AI-assisted reports and config assistance with explicit review and cost
  gates;
- managed study execution, campaign/variant orchestration, dashboards, team
  review and signoff, governed templates, retention, and customer-data policy;
- frontier-model evaluation harnesses, hidden evaluator truth, model-provider
  execution, scoring, and campaign evidence;
- custom GNC workbenches, spacecraft packages, cFS/SIL, and program-specific
  integrations;
- the private FSWDK workflow superset: Controller Bench, tuning, qualification,
  baseline promotion, packaged review evidence, and external-process candidates;
- private `agents/pro/` instructions and capability routing for Pro workflows;
- private onboarding, support, scenario migration, and customer deliverables.

The public Lambert planner is a bounded two-body trade-space tool. It does not
make general optimization, uncertainty analysis, or operational maneuver
planning public.

The public trajectory targeter is a local equality-constraint corrector. It
does not include bounds, inequality/path constraints, finite-burn optimization,
multiple shooting, collocation, multi-start/global search, campaign-scale
robustness analysis, or operational maneuver authorization. Those workflow and
optimization layers remain Pro.

The experimental `sim.trajectory_targeting` package is a separate private
research interface. Its force-configurable ONP adapter, multi-burn examples,
tests, and reference packets are excluded from the generated public repository.
It does not change the existing public targeter contract.

The private `sim.pro_trajectory_optimization` package now implements the first
of those Pro layers: bounded objectives and constraints, finite-burn mass
depletion, state-and-mass multiple shooting, deterministic multi-start and
local robustness, two open SciPy solver adapters, and content-bound replay.
This does not promote collocation, a global optimizer, campaign-scale
probability, maneuver authorization, or flight qualification. The package,
contract, examples, and tests are excluded from generated public exports, and
the public targeter code and contract remain unchanged.

The private `sim.pro_constellation_optimization` package implements the first
Pro design-search layer above the public constellation evaluator. It generates
a bounded declared discrete grid, supports exhaustive or deterministic seeded
selection, adds analyst-declared cost feasibility and Pareto ranking, retains
resumable content-bound checkpoints, and authoritatively reevaluates every
promoted design before atomic evidence publication. It is separately gated by
`constellation_optimization`; its discoverable capability ID is
`constellation_design.optimization`. The implementation, contract, example,
tests, and guide are excluded from generated public exports. This does not add
continuous ground-site placement, crosslink routing, station capacity,
uncertainty campaigns, calibrated cost, or global-optimum proof.

The canonical commercial family for these workflows is **OEL Pro Orbit
Determination**, with product-family identifier `orbit_determination`. The
existing `tracking_od` feature key remains the entitlement and license-file
identifier for v0.29 compatibility. Within that family,
`orbit_determination.reduced_tracking` identifies reduced ground/optical
tracking OD and `orbit_determination.ilrs_slr` identifies ILRS/SLR OD. These
are separately discoverable capabilities, not separate v0.29 entitlements or
SKUs.

The private `sim.pro_tracking_od` package extends the existing Pro sequential
EKF/RTS owner to reduced ground topocentric range/range-rate and inertial
optical RA/Dec observations. It preserves an untouched holdout, propagates the
filtered state and covariance to each holdout epoch, and retains residual/NIS,
covariance, rejection, and bounded maneuver-change evidence. It is separately
entitled from public OEL by `tracking_od`, excluded from generated public exports, and does not
claim raw Doppler reduction, calibrated predicted accuracy, custody, or
operational maneuver attribution.

The private `sim.pro_slr_od` package adds the entitled real-data composition
for ILRS CRD v2 passive two-way normal points. It safely captures the source
bytes, performs an OGP mean-element and aggregate-station-bias fit, runs the
shared EKF/RTS owner on fit data, and retains separate validation and untouched
holdout predictions with replayable receipts. The public export continues to
exclude the workflow, contract, fixtures, tests, and guide. Existing public
standards/interchange foundations are unchanged; no public batch campaign or
real-data orchestration was added.

The public conjunction surface is a bounded two-object educational workflow.
It does not include catalog monitoring, event association, covariance
calibration or propagation, multiple/nonlinear Pc methods, constrained
avoidance optimization, full-catalog rescreening, or operational collision-
avoidance authority. Those scale, uncertainty, optimization, and governed
workflow layers remain Pro.

The public Hosted OEL client is an access and verification surface. It does
not make hosted executors, Pro source, private worker images, campaigns, or
review-packaging implementations part of the public core.

Public coverage sensitivity compares explicitly supplied deterministic
coverage products; it does not make Pro campaign orchestration public. Public
bounded tasking, constellation aggregation, and exact multi-asset scheduling
do not include managed or rolling-horizon replanning, customer catalogs,
operational-scale optimization, crosslink routing, weather/interference
services, or proprietary calibrated equipment data. The separate public power
workflow can replay one selected schedule against a supplied orbit and lumped
battery; it does not add battery state to the scheduling solver itself.

The public `spacecraft-power` workflow is a retained-history feasibility
analysis over one supplied ECI history and declared load timeline. It does not
run the ONP spacecraft resource model or add battery state to the scheduling
solver itself. The separate public ONP `specs.thermal` and `specs.power`
models integrate one body thermal node, fixed ideal solar panels, and a bounded
battery for supported Earth-centered satellite runs. Their resource histories
are simulation truth, not sensor measurements.

Those ONP resource models do not include a multi-node thermal network, panel
temperature or panel-to-body conduction, temperature-dependent cell behavior,
battery aging, detailed bus/regulator topology, brownout, or automatic load
shedding. They are not supported for OGP or CR3BP and do not provide thermal or
power qualification. Higher-fidelity spacecraft resource models and
calibrated/uncertainty-driven resource analysis remain outside the current
public workflow.

The public headless ground segment uses geometric access and explicit service
outages, seeded station measurements, delayed packet delivery, ideal onboard
resource telemetry, and a two-body ground estimator initialized by an explicit
prior. It has no RF/link-budget threshold, weather, bandwidth, packet-loss,
command-execution, onboard-recorder, or out-of-sequence smoothing model, and is
not an operational tracking qualification.

The public orbit-lifetime surface is one bounded deterministic ONP propagation,
not a calibrated lifetime, compliance, or reentry-risk service. Managed current
or historical weather ingestion, density and ballistic-coefficient calibration,
uncertainty and Monte Carlo, campaign/constellation trades, customer models,
operational-scale performance, compliance packages, and qualification evidence
remain Pro or future.

The public optical collection workflow is a deterministic requirement screen,
not calibrated payload qualification. It does not include weather, clouds,
terrain occlusion, atmospheric/radiometric performance, exact swept footprints,
operational multi-observation replanning, advanced radar, or collection
authority. The separate public bounded scheduler accepts already-proven
opportunities and provides event-based resource and delivery accounting;
governed data, calibration, operational scale, and workflow remain Pro.

The public TDM tracking-OD surface is one bounded, analyst-supplied dataset and
one inspectable estimator run. It does not include live or bulk ingestion,
customer-data governance, raw radiometric reduction, Doppler, light-time/media
or transponder calibration, association or custody, multi-dataset campaigns,
calibrated predicted orbit accuracy, tracking-schedule optimization, or
operational OD authority. Those governed data, calibration, scale, and
workflow layers remain Pro.

The public study lifecycle is a local provenance and claims layer over already
completed supported evidence. It does not execute or substitute for domain
physics, manage queues or recovery, migrate old records, authorize decisions,
or provide a collaborative analyst workbench. Those scale, governance,
collaboration, and packaged-review layers remain Pro or future work.

Frontier-model evaluation source, configs, tests, provider integrations, and
generated evidence remain private. Public release notes may describe the
boundary and compatibility work, but that does not promote the evaluator into
the public core.

Public agents may expose recommendation-only product discovery from
`oel://analysis/workflows/v1` when a user's request materially exceeds the
matching public workflow. Every such Pro entry is labeled **coming soon and not
currently available for purchase or execution**, has no MCP execution tools,
and includes a public fallback. This metadata does not grant entitlement,
promise price or launch timing, or promote private implementation into the
public core.

Public numeric capabilities include their corresponding Rust implementation
source. Shipping a compiled dependency alone does not satisfy the public-source
boundary. Pro-only workflow and native owners remain outside the public export.

## Promotion Rule

Promote a capability when it improves adoption, inspectability, education, or
trust in the core and can be published safely with honest limits.

Keep a capability private when its primary value is expert labor reduction,
analysis at scale, customer-specific integration, proprietary data handling, or
packaged mission/release evidence.

Public-safe validation evidence belongs public when it is reproducible,
redistributable, and tied to a bounded claim. Customer data, proprietary
reference material, generated private reports, and program-specific evidence do
not.

## Repository And Export Model

The private repository is the source of truth. Public releases are generated,
not hand-curated:

1. Develop and validate changes privately.
2. Deliberately promote public paths in
   `docs/operations/public_surface_manifest.yaml`.
3. Keep private surfaces in the defense-in-depth exclusions in
   `docs/operations/public_export_exclude.txt`.
4. Generate the export with `tools/export_public.py`.
5. Run `tools/check_public_export.py` and the release gate.
6. Review the generated public diff before opening or updating a public PR.

Anything outside the positive public manifest is private by default. Never push
the private working tree directly to the public repository.

## Access Model

Public examples require neither Pro nor hosted AI accounts. OEL Pro is not
currently offered for purchase or public self-service execution. Any future
pilot access will be announced separately with its supported workflows,
versions, license terms, evidence expectations, support boundaries, and
applicable security, procurement, data-handling, and export constraints.

For detailed private workflows, use the Pro User Guide in the full workspace.
For public limitations, use [Known Limitations](known-limitations.md).

## Hosted availability

**Closed alpha: access is not publicly available.** Public OEL remains free to
run locally without a Hosted account. The bundled client and package validator
do not grant service access. Only invited operators with a configured profile
can request alpha offers; the alpha uses simulated settlement, not real payments.
