# Product Inventory

This inventory summarizes the major workflow, controller, and mission surfaces
across the public core and Pro workspace. It answers both "what exists?" and
"what is actually packaged here?" before a user starts reading Python modules
or reverse-engineering scenario YAML.

Use this page with:

- [Scenario YAML](scenario-yaml.md) for pointer syntax.
- the private Controller Bench guide for comparative evaluation workflows;
- [Controller Naming Conventions](project/controller_naming_conventions.md) for
  naming rules when adding or renaming controllers.
- [Built-In Reference GNC Library](reference-gnc.md) for maintained controller
  and command-module behavior and support posture.
- [Public Core And Pro Boundary](public-vs-pro.md) for public/pro packaging
  posture.

Inventory labels:

- **Flagship**: curated, product-facing path for demonstrations or evidence.
- **Reference**: supported baseline or reusable building block.
- **Workbench**: useful for engineering studies, tuning, wrapping, or custom
  scenarios.
- **Experimental**: included for exploration; validate before relying on it.
- **Compatibility**: retained for older configs or migration paths.

Packaging and maturity are separate. **Public** means the implementation and
documented workflow ship in the open-core export. **Pro** means the capability
exists in the private product workspace even when a related public primitive is
available.

A working implementation remains **Experimental** until it has a documented
interface, deterministic tests, a maintained workflow, bounded claims, and an
explicit packaging decision.

Packaging and technical maturity do not establish commercial availability.
Recommendation-only Pro metadata is labeled coming soon until a separate
release makes purchase, entitlement, and execution available; it must not be
used to imply a launch date, price, or operational qualification.

## Capability Status

| Capability | Packaging | Status | Evidence entry point | Primary limit |
| --- | --- | --- | --- | --- |
| Scenario YAML, CLI, and Python API | Public | Flagship | [Quickstart](quickstart.md), engine and scenario contracts | Deterministic serial object stepping for public single-scenario execution; pre-1.0 non-contract fields may still evolve. |
| ONP numerical propagation | Public | Reference | [Physics Model Reference](physics-models.md), public scenarios | Fidelity depends on the configured force models and validation envelope. |
| Earth precision forces | Public | Experimental | [Optional Earth precision forces](orbit-precision-forces.md), public synthetic demonstration and model-specific regression checks | Albedo/IR, Schwarzschild, solid Earth tides and ocean tidal gravity; explicit data/frame requirements, optional Rust kernels, no operational qualification claim. |
| OGP-SGP4/SDP4 passive propagation | Public | Reference | TLE examples, OGP reference suites, frame provenance | Passive catalog-style propagation; not an operational catalog service. |
| CCSDS and canonical frame/time interchange | Public | Experimental | [OEM](ccsds-oem.md), [OPM/OMM](ccsds-odm.md), [canonical frame/time](frame-time.md), public fixtures | Bounded KVN profiles and explicit named transforms only; not complete CCSDS coverage, covariance calibration, or operational interchange qualification. |
| Modular external data IO | Public foundation; Pro review fitting | Experimental | [Data IO](data-io.md), `oel io formats`, format owners and review post-processors | Bounded local format profiles, shared object composition, and exports from completed read-only evidence. Format conversion and materialization do not execute simulations or fit orbits; review-to-OGP/TLE fitting remains Pro. No complete-standards or operational-accuracy claim. |
| Coverage and directed-link analysis | Public | Experimental | [Scenario workflow](coverage-link-scenario-analysis.md), public example, contracts, and [programmatic acceptance](validation-coverage-link-programmatic.md) | Deterministic sampled engineering analysis; independent external validation, calibrated hardware/environment models, and operational assurance remain outside the claim. |
| Optical collection-opportunity analysis | Public | Experimental | [Collection workflow](collection-opportunity-analysis.md), public example and regression suite | One WGS84 target and one hard-FOV payload with transparent first-order screening; no calibrated payload, weather, terrain, or operational collection claim. |
| Conjunction assessment | Public | Experimental | [Conjunction workflow](conjunction-assessment.md), CDM and synthetic assessment fixtures | Bounded two-object geometry, one educational 2D Pc, and small explicit rescreens; no catalog monitoring, calibrated covariance, or maneuver authority. |
| Reduced-geometric CCSDS TDM tracking OD | Public | Experimental | [Tracking-OD workflow](tracking-od.md), synthetic fit/holdout example | One analyst-declared reduced-geometric AZEL/range dataset and one batch fit; no raw radiometric reduction, calibrated predicted accuracy, association, custody, or operational OD. |
| Bounded multi-asset mission scheduling | Public | Workbench | [Mission-scheduling workflow](mission-scheduling.md), exact and source-aware replay fixtures | Exact subset search over at most 18 supplied opportunities; no opportunity generation, battery/thermal state, routing, rolling-horizon replanning, or operational-scale optimization. |
| Bounded constellation and ground-network design | Public | Workbench | [Constellation-design workflow](constellation-design.md), Walker/network example and authoritative replay | Exact evaluation of at most eight explicit circular Walker/shell candidates with ideal-nadir coverage and same-epoch free-space ground links; no automatic search, capacity/contention, crosslinks, uncertainty, calibrated availability, or operational qualification. |
| Spacecraft-power feasibility analysis | Public | Experimental | [Power workflow](spacecraft-power.md), schedule-coupled canonical example | One retained orbit and lumped battery/load timeline; separate from scenario resource state and not detailed EPS, thermal, uncertainty, qualification, or operational power authority. |
| ONP spacecraft thermal and power resources | Public | Experimental | [Spacecraft resources](models/spacecraft-resources.md), resource demo and review-history fixtures | Earth-centered ONP satellites; one body thermal node, fixed ideal panels, and a bounded battery; no detailed EPS, degradation, load shedding, resource qualification, or OGP/CR3BP support. |
| Headless ground segment | Public | Experimental | [Ground segment](models/ground-segment.md), delayed tracking/telemetry demo and review tables | Geometric availability, explicit outages, delayed packets, ideal resource telemetry, and two-body estimation from a supplied prior; no RF assurance, command execution, out-of-sequence smoothing, or operational qualification. |
| Deterministic orbit-lifetime analysis | Public | Experimental | [Orbit-lifetime workflow](orbit-lifetime.md), frozen-atmosphere comparison example | One bounded ONP drag-decay case with frozen inputs; not calibrated lifetime, compliance, surviving-debris risk, custody, or operational authority. |
| Integrated study lifecycle | Public | Workbench | [Study lifecycle](study-lifecycle.md), six-domain registry and canonical studies | Local content-bound records, claims, inspection, replay, and comparison over completed evidence; no managed execution, domain-physics replay, team workflow, or authorization. |
| Governed communications engineering | Pro | Experimental, unreleased private-worktree surface | Private Pro communications workflow, governed profiles, validation manifest, and content-bound evidence bundle | Enforced by the `pro_communications` feature at direct workflow entry points. Source must be tracked and included in a future Pro package candidate before distribution; included terminals/sites are illustrative, and measured RF, current weather/interference, packet assurance, and operational availability are not claimed. |
| GNC v2 complete-stack runtime and profiles | Public | Mixed | [GNC v2 evidence](gnc-v2-evidence.md), [flight-software profiles](flight-software-profiles.md) | All 18 exact profiles declare Supported within bounded simulation envelopes; effective maturity requires current qualification evidence (see profile discovery); underlying stacks and arbitrary custom compositions remain Experimental. |
| Public FSW Authoring Kit | Public | Workbench | [Public FSW Authoring](fsw-authoring.md), content-bound validation/test/smoke receipts | ADCS/RPO Python-stack authoring and one deterministic serial smoke only; no comparison, tuning, qualification, external process, or cFS/SIL workflow. |
| Lambert orbit-transfer planning | Public | Workbench | `configs/orbit_transfer_planner_demo.yaml`, review-store candidate tables | Bounded two-body grid search, not operational or globally optimal planning. |
| Event-driven trajectory targeting | Public | Workbench | [Trajectory-targeting workflow](trajectory-targeting.md), Hohmann and open-reference fixtures | Local single-shooting equality correction with authoritative ONP repropagation; no bounds, inequalities, finite burns, multiple shooting, global search, or maneuver authority. |
| General trajectory optimization | Pro | Workbench | Private Pro trajectory-optimization guide, retained three-problem workflow pack, analytic and independent-solver tests | Local constrained nonlinear optimization with mandatory ONP repropagation; no collocation, global-optimum proof, calibrated probability, maneuver authority, or flight qualification. |
| OEL Pro Constellation Design | Pro | Workbench | Private Pro constellation-optimization guide, exhaustive example, deterministic/Pareto/resume/replay tests | Bounded discrete orbit and catalog-site search over the public evaluator with promoted-design reevaluation; no continuous siting, routing/capacity, uncertainty campaigns, calibrated cost, global-optimum proof, or operational qualification. |
| Review store, queries, and plotting | Public | Reference | [Review Store](review-store.md), [Plotting](plotting.md) | Only recorded evidence can be queried or plotted. |
| RPO Trainer | Public | Flagship | `run_game.py`, game configs and debriefs | Educational; not operational training qualification. |
| Rocket ascent, aero, and re-entry | Public | Workbench | public configs, rocket contract, model docs | First-pass engineering models with bounded validation evidence. |
| ML/RL wrappers | Public | Experimental | [ML/RL Policy Contracts](ml-rl-contracts.md) | Optional dependencies and limited reproducible benchmark coverage. |
| Campaigns, sensitivity, covariance, orbital-delivery accuracy, and controller bench | Pro | Reference | Private workflow guides and contracts | Includes rocket-driven payload delivery covariance, percentile, correction-delta-v proxy, and RPO feasibility evidence; not included as public workflow automation. |
| Governed/general ONP dynamics and ground-sensor batch orbit determination | Pro | Reference | Private OD contract, synthetic OD harness, Scale sensor-OD workflows | Broader governed datasets, systematics, campaigns, and customer workflows remain Pro; the separate bounded public TDM fit/holdout row above is not promoted into this scope. |
| OGP-SGP4/SDP4 mean-element orbit determination | Pro | Reference | Private OD contract, Phase 6 harness, Scale SGP4 sensor-OD workflow | Regime-bounded same-family L2 OD; B*/epoch gates are case-specific and operational publication remains outside the claim. |
| OGP two-way SLR normal-point orbit determination | Pro | Experimental | Private OD contract, Phase 4 external campaign, frozen two-day actual-ILRS campaign | The complete actual-data chain is replayable, but the two-day campaign failed its residual-whiteness qualification gates; covariance and predicted accuracy remain uncalibrated. |
| Initial-orbit and optical angles-only determination | Pro | Experimental | Private OD contract, Phase 5 synthetic campaign | Gibbs/Herrick-Gibbs and ambiguity-aware Gauss/Lambert initialization have bounded L2 synthetic evidence; no operational optical, association, or custody claim. |
| OEL Scale and intent-hypothesis evaluation | Pro | Experimental | Scale validation tiers, catalog/campaign and ONP-refinement queues, IHE core/observational packs, and typed handoffs | Implemented deterministic local slices with analyst review and dedicated `scale_analytics` / `intent_hypothesis` entitlement gates; no calibrated operational catalog, probability, association, custody, or subjective-intent claim. |
| AI-assisted reports and config assistance | Pro | Workbench | Private staged estimate/create workflows | Requires explicit provider, data, cost, and human-review decisions. |
| cFS/SIL common-boundary adapter | Pro/private | Experimental | Private cFS/SIL ICD and byte-level golden vectors | Excluded from the public export; prototype loopback transport and conformance envelope, not flight or real-time qualification. |
| FSW Development and Verification Kit | Pro/private | Workbench | Private FSWDK contract, Controller Bench, qualification, and packaging guides | Strict workflow superset of public authoring; private evidence automation does not turn simulation results into flight qualification. |

## GNC v2 complete stacks

Every satellite using the default `flight_software` runtime profile crosses one
`SatelliteFlightSoftware` boundary. Typed
sensor, command, load, pilot, and telemetry events enter; typed device commands
leave. Navigation, goals, constraint handling, executive logic, guidance,
control, allocation, and stack-owned recovery are internal implementation
details. A propagation-only satellite with no `flight_software` declaration
uses the v2 passive stack. An explicitly declared `runtime_profile:
trajectory_only` bypasses onboard modeling and retains only configured
deterministic dynamics. Any non-empty v1 satellite GNC field is rejected.

| Stack | Hardware currently realized by the reference factory | Included modes | Maturity |
| --- | --- | --- | --- |
| `fsw.passive` | Passive | Coast and typed evidence | Experimental |
| `fsw.attitude_reference` | Ideal torque/wrench | Quaternion hold and configured reference pointing | Experimental |
| `fsw.orbit_reference` | Ideal wrench | Stationkeeping and orbital-element feedback | Experimental |
| `fsw.rpo_reference` | Ideal wrench | RIC hold, axis approaches, waypoint, braking, and passive retreat | Experimental |
| `fsw.low_thrust_reference` | Continuous engine with body-frame gimbal commands | Low-thrust phasing | Experimental |
| `fsw.game_pilot_reference` | Ideal wrench or modeled aerodynamic effectors, by declared input profile | Pilot translation, attitude/thrust, operator input, and aerodynamic play | Experimental |

The stack catalog and component catalog are intentionally separate. Public
custom stacks can compose component math without Pro. The Pro Controller Bench
replaces and evaluates complete stack compositions; it is not a second runtime.
The exported [flight-software profiles](flight-software-profiles.md) retain the
public qualification envelope; private migration and evidence details remain in
the full workspace.

OEL also ships an initial catalog of 18 versioned
[flight-software use-case profiles](flight-software-profiles.md) across
commissioning/ADCS, absolute-orbit operations, RPO/formation applications, and
low-thrust control. Their declared Supported maturity applies only inside the
qualification envelope and is distinct from current effective maturity. Use
`python -m sim.flight_software list --kind profile --json` to inspect effective
status; missing or stale evidence yields Experimental. That maturity does not promote
any underlying stack for arbitrary compositions or parameters outside the
profile envelope.

## Mission And Dynamics Capability Inventory

These surfaces are not all "controllers", but they are user-facing product
capabilities that ship with the simulator and appear in scenario YAML, plots,
payloads, or validation workflows.

| Product surface | Config surface | Status | Primary use |
| --- | --- | --- | --- |
| Shared Vehicle Aero Contract | `objects.<id>.specs.aero` | Reference | Canonical vehicle aero properties shared by satellites, re-entry diagnostics, aero-assist studies, and rocket ascent defaults. |
| Orbit Drag Force Model | `simulator.dynamics.orbit.drag` | Reference | Enables atmosphere-relative drag acceleration using configured vehicle area/coefficient and environment density. |
| Atmospheric Lift Force Model | `objects.<id>.specs.aero.cl`, `lift_area_m2`, `lift_axis_body` plus `orbit.drag` | Workbench | First-pass attitude-coupled lift for atmospheric steering and plane-change exploration. |
| Re-entry Diagnostics | `simulator.dynamics.reentry` | Reference | Tracks entry episodes, dynamic pressure, drag g-load, heat rate, heat load, and object-level termination limits. |
| Aero-Assisted Atmospheric Pass | `configs/aero_assisted_plane_change_demo.yaml` | Workbench | Demonstrates a satellite dipping into the atmosphere, using lift-axis steering, then burning to raise altitude. |
| Rocket Ascent Aero | `objects.rocket.specs.aero` plus `simulator.dynamics.rocket.aero` | Workbench | Shared object geometry/defaults with detailed rocket coefficient refinements, max-Q/TVC diagnostics, and insertion studies. |
| Object-Based Aero/Re-entry Termination | `simulator.dynamics.reentry.termination.by_object` and `simulator.termination.by_object` | Reference | Lets rockets, disposed stages, satellites, and atmospheric test articles coexist with different stop conditions. |

Canonical vehicle aero fields live under `objects.<id>.specs.aero`:
`reference_area_m2`, `drag_area_m2`, `lift_area_m2`, `cd`, `cl`,
`nose_radius_m`, `reference_length_m`, `lift_axis_body`/`lift_vector_body`, and
`cp_offset_body_m`. Use `simulator.dynamics.orbit.drag` to enable aerodynamic
forces, `simulator.dynamics.reentry` for diagnostics and limits, and
`simulator.dynamics.rocket.aero` only for detailed rocket coefficient
refinements. Flat aliases such as `specs.drag_area_m2`, `specs.cd`, and
`specs.nose_radius_m` remain compatibility inputs.

## Estimation And Knowledge

The first four live-knowledge rows below describe the pre-v2 component library,
not a supported v2 satellite configuration surface. Their filter mathematics
remain available for composition work, but a v2 stack must expose them through
typed measurement adapters before claiming them as onboard capability. Batch OD
workflows remain separate ground/offline estimation products.

| Product surface | Packaging | Config/API surface | Status | Primary use |
| --- | --- | --- | --- | --- |
| ECI State EKF Knowledge | Public | `knowledge.estimation.type: ekf` | Reference | Live target-state belief from noisy ECI state or relative measurements. |
| EKF Maneuver Detection | Public | `knowledge.estimation.maneuver_detection` / `EKFManeuverDetector` | Reference | Innovation/NIS persistence gate for EKF knowledge tracks; reports suspect/confirmed maneuver evidence in knowledge consistency summaries. |
| Measured State Knowledge | Public | `knowledge.estimation.type: measured_state` | Reference | Trust the latest full-state measurement while publishing a `StateBelief`. |
| HCW Relative EKF Knowledge | Public | `knowledge.estimation.type: relative_hcw_ekf` / `HCWRelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for circular-chief, small-separation RPO scenarios; public validation covers full relative-state and az/el/range/range-rate measurement cases. |
| TH Relative EKF Knowledge | Public | `knowledge.estimation.type: relative_th_ekf` / `THRelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for eccentric two-body-chief, small-separation RPO scenarios using numerically integrated TH linear dynamics; public validation covers full relative-state and az/el/range/range-rate measurement cases. |
| YA STM Relative EKF Knowledge | Public | `knowledge.estimation.type: relative_ya_ekf` / `YARelativeEKFEstimator` | Reference | Live rectangular-RIC relative-state estimation for eccentric two-body-chief RPO using the closed-form Yamanaka-Ankersen anomaly-domain STM mapped into OEL's km/km/s RIC state; public validation compares it against HCW, TH-integrated, and ECI EKF rows. |
| Full-state EKF Innovation Recovery | Public | `knowledge.estimation.ekf.innovation_recovery` / `InnovationRecoveryConfig` | Experimental | Off-by-default causal consistency or persistence recovery for HCW, SS-J2, TH, YA and absolute orbit EKFs. Full position/velocity observations only; recovery evidence is separate from maneuver identification. See [usage and limits](innovation-recovery.md). |
| ONP Dynamics Batch Orbit Determination | Pro | `sim.estimation.orbit_determination` / `solve_dynamics_orbit_determination` | Reference | Fits initial ECI Cartesian state and identifiable drag/Cd/SRP scales with ONP in the residual loop; supports full observation covariance, explicit priors, robust/rejection policy, observability diagnostics, grouped residual audits, fit/holdout evidence, and materialized scenarios. |
| ONP Native Ground-Sensor Orbit Determination | Pro | `sim.estimation.ground_station_od` / `sim.scale fit-ground-sensor-od` | Reference | Fits initial ECI Cartesian state against individual/combined azimuth, elevation, range, and range-rate rows; optional station/shared biases and clock terms carry priors and identifiability evidence, with elevation weighting, apparent-elevation refraction, station holdout/exclusion, and queryable review tables. |
| OGP Mean-Element Orbit Determination | Pro | `sim.estimation.ogp_od` plus legacy `sim.estimation.sgp4_od` / Scale commands | Experimental reference | Fits native TEME PV across named OGP-SGP4 and OGP-SDP4 regimes with automatic nonsingular/classical parameterization, fit/holdout evidence, and gated B*/epoch trials. Legacy fixed-epoch SGP4 PV and ground-sensor routes remain supported; Phase 6 is bounded same-family synthetic OD, not precise truth or operational TLE publication. |
| Pro Tracking-Data Sequential OD | Pro | OEL Pro Orbit Determination / `orbit_determination.reduced_tracking`; `python -m sim.pro_tracking_od` | Experimental reference | Uses the shared v0.29 `tracking_od` entitlement for EKF/RTS over reduced ground range/range-rate or inertial optical RA/Dec rows with mandatory untouched holdout covariance prediction; no raw radiometric reduction or calibrated predicted-accuracy claim. |
| Pro ILRS/SLR Orbit Determination | Pro | OEL Pro Orbit Determination / `orbit_determination.ilrs_slr`; `python -m sim.pro_slr_od` | Experimental reference | Uses the shared v0.29 `tracking_od` entitlement for CRD v2 OGP batch fit plus fit-only EKF/RTS and distinct validation/untouched-holdout predictions. The frozen actual-data campaign replayed but failed residual-whiteness gates; no calibrated covariance, predicted accuracy, or geodetic-quality claim. |

## Orbital component library

These implementations are retained as component mathematics and historical
evidence. They are not top-level satellite scenario fields and their old
Reference/Workbench labels do not promote a complete v2 stack. Use the
component catalog when composing a custom stack, then test that composition
through the common boundary.

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Zero Controller | `ZeroController` | `sim.control.orbit.zero_controller` | Reference | Coast/no-thrust baseline and config smoke tests. |
| RIC_PD Transfer | `RICPDTransferController` | `sim.control.orbit.ric_pd` | Flagship | 10 km-class RIC-frame RPO transfer with guided coast, correction burns, and terminal PD cleanup. |
| RIC PD Hold (HCW compatibility name) | `HCWPDController` | `sim.control.orbit.hcw_pd` | Workbench | Rectangular-RIC PD hold with optional HCW feedforward; the class name is retained until the product-facing RIC-hold migration is complete. |
| HCW LQR | `HCWLQRController` | `sim.control.orbit.lqr` | Reference | Linear HCW rendezvous/control baseline for closed-loop RPO examples and benchmarks. |
| SS-J2 LQR | `SSJ2LQRController` | `sim.control.orbit.lqr` | Workbench | Near-circular chief relative-orbit LQR using the shared homogeneous Schweighart-Sedwick averaged-J2 model. |
| HCW LQR, No Radial Burn | `HCWNoRadialLQRController` | `sim.control.orbit.lqr_no_radial` | Workbench | HCW LQR variant that suppresses radial-axis burns for constrained RPO studies. |
| HCW Manual Gain, No Radial Burn | `HCWNoRadialManualController` | `sim.control.orbit.lqr_no_radial` | Workbench | Manually supplied gain matrix for no-radial-burn tuning experiments. |
| SS-J2 LQR, No Radial Burn | `SSJ2NoRadialLQRController` | `sim.control.orbit.lqr_no_radial` | Workbench | SS-J2 relative LQR variant constrained to in-track and cross-track acceleration. |
| HCW Curvilinear Input Variant | `HCWCurvInputRectOutputController` | `sim.control.orbit.lqr_curv_variant` | Workbench | HCW controller variant for curvilinear-RIC inputs with rectangular-RIC output behavior. |
| Curvilinear RIC PD | `CurvilinearRICPDController` | `sim.control.orbit.curv_pd` | Workbench | PD control in curvilinear RIC coordinates, with conversion to rectangular RIC/ECI commands. |
| Relative Orbit MPC | `RelativeOrbitMPCController` | `sim.control.orbit.relative_mpc` | Experimental | General relative-orbit model-predictive control experiments. |
| HCW Relative MPC | `HCWRelativeOrbitMPCController` | `sim.control.orbit.hcw_mpc` | Experimental | HCW-based MPC studies over a finite horizon. |
| HCW Relative MPC, In/Cross Track | `HCWInTrackCrossTrackMPCController` | `sim.control.orbit.hcw_mpc` | Experimental | MPC variant constrained to in-track and cross-track control axes. |
| SS-J2 Relative MPC | `SSJ2RelativeOrbitMPCController` | `sim.control.orbit.hcw_mpc` | Experimental | Finite-horizon relative MPC using the shared near-circular SS-J2 prediction model. |
| SS-J2 PD | `SSJ2PDController` | `sim.control.orbit.hcw_pd` | Workbench | Convenience rectangular-RIC PD controller configured for SS-J2 feedforward. |
| RMOE If-Then | `RMOEIfThenController` | `sim.control.orbit.rmoe` | Workbench | Rule-based relative mean orbital-element/NMC targeting with priority logic and drift limiting. |
| Scheduled Impulse | `ScheduledImpulseController` | `sim.control.orbit.scheduled_impulse` | Reference | Deterministic delayed finite-duration impulse for validation and maneuver-detection proof scenarios. |
| Atmospheric Pass | `AtmosphericPassController` | `sim.control.orbit.aero_assist` | Workbench | Timed coast/raise-burn controller for aero-assisted atmospheric-pass demos. |
| Stationkeeping | `StationkeepingController` | `sim.control.orbit.baseline` | Reference | ECI state-hold or simple target-state feedback. |
| SMA/Ecc Feedback | `SemiMajorAxisEccentricityController` | `sim.control.orbit.baseline` | Reference | Low-thrust semi-major-axis and eccentricity regulation. |
| COE Feedback | `OrbitalElementsFeedbackController` | `sim.control.orbit.baseline` | Reference | Classical-orbital-element tracking across selected elements. |
| Safety Barrier | `SafetyBarrierController` | `sim.control.orbit.baseline` | Experimental | Repulsive keep-out style safety behavior inside a configured radius. |
| Risk Threshold | `RiskThresholdController` | `sim.control.orbit.baseline` | Experimental | Switches between nominal and evasive controllers using a user-supplied risk function. |
| RIC Relative Hold | `RICRelativeHoldController` | `sim.control.orbit.reference_rpo` | Reference | Hold a configured rectangular-RIC position and rate. |
| V-Bar Approach | `VBarApproachController` | `sim.control.orbit.reference_rpo` | Reference | Rate-limited in-track approach with terminal slowdown. |
| R-Bar Approach | `RBarApproachController` | `sim.control.orbit.reference_rpo` | Reference | Rate-limited radial approach with terminal slowdown. |
| C-Bar Approach | `CBarApproachController` | `sim.control.orbit.reference_rpo` | Reference | Rate-limited cross-track approach with terminal slowdown. |
| RIC Waypoint Path | `RICWaypointController` | `sim.control.orbit.reference_rpo` | Reference | Tolerance-gated rectangular-RIC waypoint sequence. |
| Keep-Out Standoff | `KeepOutStandoffController` | `sim.control.orbit.reference_rpo` | Reference | Spherical geometric keep-out response and standoff recovery. |
| Passive-Safe Retreat | `PassiveSafeRetreatController` | `sim.control.orbit.reference_rpo` | Reference | Acquire an outward relative drift rate, then coast. |
| Terminal Braking | `TerminalBrakingController` | `sim.control.orbit.reference_rpo` | Reference | Closing-rate-limited terminal relative-state settling. |
| RIC Flyaround | `RICFlyaroundController` | `sim.control.orbit.reference_rpo` | Reference | Cyclic polygonal inspection flyaround in an RIC plane. |
| Low-Thrust Phasing | `LowThrustPhasingController` | `sim.control.orbit.reference_rpo` | Workbench | Low-authority along-track relative phasing baseline. |
| Plane-Change Trim | `PlaneChangeTrimController` | `sim.control.orbit.reference_rpo` | Workbench | Cross-track relative-position/rate trim. |
| HCW Rendezvous Planner | `HCWRendezvousPlannerController` | `sim.control.orbit.reference_rpo` | Reference | Closed-form linear velocity acquisition with finite-burn realization. |
| Proportional Navigation | `ProportionalNavigationController` | `sim.control.orbit.reference_rpo` | Workbench | Target-directed or target-opposed local RIC proportional navigation. |

Actuator-aware orbital wrappers are covered in more detail in
[Actuators](actuators.md), but they are part of the controller inventory:

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| RCS Allocation Aware | `RCSAllocationAwareController` | `sim.control.orbit.rcs_allocator` | Workbench | Previews/caps desired commands against configured RCS cluster authority. |
| Electric Propulsion | `ElectricPropulsionController` | `sim.control.orbit.electric_propulsion` | Workbench | Caps a base controller by electric-propulsion thrust, duty-cycle, and power limits. |
| Gimbaled Thruster | `GimbaledThrusterController` | `sim.control.orbit.gimbaled_thruster` | Workbench | Suppresses thrust directions outside configured gimbal authority. |

Additional orbital-control helpers, such as `PredictiveBurnScheduler`,
`OrbitalAttitudeManeuverCoordinator`, and the HCW transfer solvers, are library
building blocks rather than top-level user-selectable controllers.

## Attitude component library

These implementations are candidate component mathematics. They are not
configured through `attitude_control` in v2, and snap/surrogate state-changing
controllers are not physical GNC paths.

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Zero Torque | `ZeroTorqueController` | `sim.control.attitude.zero_torque` | Reference | No-torque baseline and attitude-disabled smoke paths. |
| Quaternion PD | `QuaternionPDController` | `sim.control.attitude.baseline` | Reference | General quaternion attitude hold/recovery with torque saturation. |
| Reaction Wheel PD | `ReactionWheelPDController` | `sim.control.attitude.baseline` | Reference | Reaction-wheel allocated quaternion PD control. |
| Reaction Wheel PID | `ReactionWheelPIDController` | `sim.control.attitude.baseline` | Workbench | Reaction-wheel PD with integral correction for persistent errors. |
| Small-Angle LQR | `SmallAngleLQRController` | `sim.control.attitude.baseline` | Reference | Linearized attitude-control baseline. |
| RIC Frame PD | `RICFramePDController` | `sim.control.attitude.ric_pd` | Reference | RIC-frame attitude hold using PD feedback. |
| RIC Frame PID | `RICFramePIDController` | `sim.control.attitude.ric_pid` | Workbench | RIC-frame attitude hold with integral action. |
| RIC Frame LQR | `RICFrameLQRController` | `sim.control.attitude.ric_lqr` | Reference | RIC-frame linear attitude-control baseline. |
| ECI Detumble PD | `ECIDetumblePDController` | `sim.control.attitude.detumble_pd` | Reference | Rate damping and detumble behavior in an ECI-oriented attitude path. |
| RIC Detumble PD | `RICDetumblePDController` | `sim.control.attitude.detumble_pd` | Reference | RIC-oriented detumble behavior for proximity operations. |
| Atmospheric Lift Axis | `AtmosphericLiftAxisController` | `sim.control.attitude.aero_assist` | Workbench | Points a configured body lift axis along a requested RIC lift direction for aero-assisted passes. |
| Snap Attitude | `SnapAttitudeController` | `sim.control.attitude.snap` | Workbench | Direct/snap-style attitude target behavior for fast scenario construction. |
| Snap And Hold RIC | `SnapAndHoldRICAttitudeController` | `sim.control.attitude.snap_hold` | Workbench | Snap to a RIC attitude and then hold it. |
| Surrogate Snap ECI | `SurrogateSnapECIController` | `sim.control.attitude.surrogate_snap` | Workbench | Fast surrogate attitude slew/recovery model in ECI. |
| Surrogate Snap RIC | `SurrogateSnapRICController` | `sim.control.attitude.surrogate_snap` | Workbench | Fast surrogate attitude slew/recovery model in RIC. |
| Detumble Then Slew | `DetumbleThenSlewController` | `sim.control.attitude.switching` | Workbench | Switches between detumble and nominal slew/hold behavior. |
| Magnetorquer B-dot | `MagnetorquerBdotController` | `sim.control.attitude.bdot_magnetorquer` | Workbench | B-field-aware detumble controller for magnetorquer studies. |
| Wheel Desaturation | `WheelDesaturationController` | `sim.control.attitude.wheel_desaturation` | Workbench | Momentum unload torque request from wheel momentum state. |
| CMG Steering | `CMGSteeringController` | `sim.control.attitude.cmg_steering` | Workbench | Caps/wraps a base attitude controller by simplified CMG authority. |
| Thrust Align | `ThrustAlignController` | `sim.control.attitude.reference_pointing` | Reference | Point a configured body thrust axis at a supplied burn direction. |
| Target Track | `TargetTrackController` | `sim.control.attitude.reference_pointing` | Reference | Point a configured boresight at a supplied target position. |
| Nadir Pointing | `NadirPointingController` | `sim.control.attitude.reference_pointing` | Reference | Point a configured body axis toward Earth center. |
| Velocity Pointing | `VelocityPointingController` | `sim.control.attitude.reference_pointing` | Reference | Prograde or retrograde body-axis pointing. |
| Sun Pointing | `SunPointingController` | `sim.control.attitude.reference_pointing` | Reference | Point a configured body axis at a supplied Sun direction. |
| RIC Axis Pointing | `RICAxisPointingController` | `sim.control.attitude.reference_pointing` | Reference | Point a configured body axis along R, I, C, or a mixed RIC direction. |

`PoseCommandGenerator` is an attitude command helper used by mission logic for
sun-track, spotlight, RIC-pointing, and target-facing commands. It is not a
standalone closed-loop controller.

## Legacy mission component inventory

The tables below inventory pre-v2 mission components for archaeology and rocket
ownership. Satellites do not execute these dictionary-producing modules. New
satellite mission behavior belongs inside a complete stack executive and typed
mission load; no Compatibility satellite module is a released runtime path.

Historically, mission logic was split into three related surfaces:

- **Mission strategies** decide intent: pursue, evade, hold, inspect,
  stationkeep, defend, or select a rocket orbital goal.
- **Mission execution modules** convert intent into integrated commands:
  pointing, predictive burns, gated burns, impulses, direct commands, or safe
  hold.
- **Mission modules** combined behavior into one plugin. These divisions remain
  relevant to rockets until their later overhaul, not to v2 satellites.

### Mission Strategies

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Pursuit | `PursuitMissionStrategy` | `sim.mission.modules` | Reference | Generate target-directed pursuit intent from object knowledge or blind fallback direction. |
| Evade | `EvadeMissionStrategy` | `sim.mission.modules` | Reference | Generate target-opposed evasive intent from object knowledge or blind fallback direction. |
| Hold | `HoldMissionStrategy` | `sim.mission.modules` | Reference | Hold ECI/RIC attitude, sun-track, spotlight, or sensing-oriented pointing intent. |
| Desired State | `DesiredStateMissionStrategy` | `sim.mission.modules` | Reference | Request an explicit or target-derived ECI desired state. |
| Station Keep | `StationKeepMissionStrategy` | `sim.mission.modules` | Reference | Maintain a desired relative RIC state around a target. |
| COE Station Keep | `OrbitalElementsStationKeepMissionStrategy` | `sim.mission.modules` | Reference | Maintain a target set of classical orbital elements at current true anomaly. |
| COE Tracking | `OrbitalElementsTrackingMissionStrategy` | `sim.mission.modules` | Reference | Track selected orbital elements with feedback acceleration intent. |
| Inspect | `InspectMissionStrategy` | `sim.mission.modules` | Workbench | Hold a relative inspection geometry while pointing at the target. |
| Defensive | `DefensiveMissionStrategy` | `sim.mission.modules` | Workbench | Burn along a fixed RIC axis or away from a known chaser. |
| Safe Hold | `SafeHoldMissionStrategy` | `sim.mission.modules` | Reference | Zero-thrust safe attitude hold or sun-track fallback. |
| Mission Executive | `MissionExecutiveStrategy` | `sim.mission.modules` | Reference | Validated priority mode machine with range, fuel, absolute-time, and mode-elapsed triggers. |
| Rocket Pursuit | `RocketPursuitMissionStrategy` | `sim.mission.modules` | Workbench | Select a rocket pursuit orbital goal. |
| Rocket Predefined Orbit | `RocketPredefinedOrbitMissionStrategy` | `sim.mission.modules` | Workbench | Select a predefined rocket target altitude/eccentricity goal. |
| Rocket Mission Strategy | `RocketMissionStrategy` | `sim.mission.modules` | Compatibility | Older combined rocket goal and launch-timing wrapper. |

### Mission Execution Modules

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Controller Pointing | `ControllerPointingExecution` | `sim.mission.modules` | Reference | Run orbit/attitude controllers, set attitude target for thrust alignment, and gate burns by pointing error. |
| Predictive Burn | `PredictiveBurnExecution` | `sim.mission.modules` | Workbench | Plan burns ahead of time, slew toward required attitude, then fire when alignment and timing allow. |
| Integrated Command | `IntegratedCommandExecution` | `sim.mission.modules` | Reference | Execute orbit-controller burns only when attitude alignment is satisfied. |
| Direct Integrated | `DirectIntegratedExecution` | `sim.mission.modules` | Reference | Pass direct thrust/torque intent through the integrated command path. |
| Impulsive | `ImpulsiveExecution` | `sim.mission.modules` | Workbench | Pulse burn commands on a configured cadence with optional attitude gating. |
| Budgeted End State | `BudgetedEndStateExecution` | `sim.mission.modules` | Workbench | Convert desired end-state velocity error into budgeted thrust-limited or impulsive maneuvers. |
| Safe Hold | `SafeHoldExecution` | `sim.mission.modules` | Reference | Force zero thrust while allowing attitude hold control. |
| Rocket Go Now | `RocketGoNowExecution` | `sim.mission.modules` | Reference | Immediately authorize launch. |
| Rocket Go When Possible | `RocketGoWhenPossibleExecution` | `sim.mission.modules` | Workbench | Authorize launch only when estimated delta-v margin is sufficient. |
| Rocket Wait Optimal | `RocketWaitOptimalExecution` | `sim.mission.modules` | Workbench | Authorize launch inside a periodic launch window. |
| Timed Finite Burn | `TimedFiniteBurnExecution` | `sim.mission.modules` | Reference | Apply an ECI or RIC acceleration over a bounded time interval. |
| One-Shot Impulse | `OneShotImpulseExecution` | `sim.mission.modules` | Reference | Apply delta-v as an explicit equivalent finite burn without state override. |
| Pulse Train | `PulseTrainExecution` | `sim.mission.modules` | Reference | Periodic duty-cycle burn sequence. |
| Slew Then Burn | `SlewThenBurnExecution` | `sim.mission.modules` | Reference | Set burn attitude, wait for alignment, then fire. |
| Burn Until Condition | `BurnUntilConditionExecution` | `sim.mission.modules` | Reference | Burn until configured time, duration, or speed criteria. |
| Coast Until Condition | `CoastUntilConditionExecution` | `sim.mission.modules` | Reference | Suppress direct thrust until time or observed-range release. |
| Command Phase Sequencer | `WaypointSequencerExecution` | `sim.mission.modules` | Reference | Advance through named duration-gated command phases. |
| Abort Safe Hold / Retreat | `AbortSafeHoldRetreatExecution` | `sim.mission.modules` | Reference | Convert abort intent into safe hold or RIC retreat. |
| Fuel Budget Gate | `FuelBudgetGateExecution` | `sim.mission.modules` | Reference | Suppress direct thrust below a configured fuel margin. |
| Keep-Out Gate | `KeepOutGateExecution` | `sim.mission.modules` | Reference | Override direct intent with an outward command inside a geometric keep-out boundary. |
| Command Replay | `CommandReplayExecution` | `sim.mission.modules` | Workbench | Replay checked command histories for regression and training. |

### Compact Mission Modules

| Product name | Class | Module | Status | Primary use |
| --- | --- | --- | --- | --- |
| Attitude Detumble Gate | `AttitudeDetumbleGateMissionModule` | `sim.mission.modules` | Workbench | Switch an attitude controller between detumble and nominal modes using rate thresholds. |
| Satellite Mission Module | `SatelliteMissionModule` | `sim.mission.modules` | Compatibility | Compact satellite behavior wrapper for coast, pursuit/evade, and attitude pointing modes. |
| Defensive RIC Axis Burn | `DefensiveRICAxisBurnMissionModule` | `sim.mission.modules` | Workbench | Defensive fixed-axis RIC burn with attitude alignment and knowledge gating. |
| Single RIC Axis Burn | `SingleRICAxisBurnMissionModule` | `sim.mission.modules` | Workbench | One-shot RIC-frame burn/slew behavior for scripted scenarios. |
| Multi RIC Axis Burn | `MultiRICAxisBurnMissionModule` | `sim.mission.modules` | Workbench | Ordered multi-axis RIC burn sequence with per-burn delta-v targets and attitude gating. |
| Scheduled Vector Burn | `ScheduledVectorBurnMissionModule` | `sim.mission.modules` | Workbench | Deterministic duration-bound ECI or RIC vector burn for interchange, OD, and scripted maneuver evidence. |
| Rocket Mission Module | `RocketMissionModule` | `sim.mission.modules` | Compatibility | Compact rocket launch authorization and goal-selection module. |
| End State Maneuver | `EndStateManeuverMissionModule` | `sim.mission.modules` | Workbench | Combined desired-state maneuver planner with integrated attitude/burn gating. |
| Integrated Command Mission | `IntegratedCommandMissionModule` | `sim.mission.modules` | Workbench | Compact module for integrated orbit/attitude command execution. |
| Predictive Integrated Command | `PredictiveIntegratedCommandMissionModule` | `sim.mission.modules` | Workbench | Compact predictive-burn variant of the integrated command path. |

## Maintainer Notes

Runtime exports in `sim/control/orbit/__init__.py`,
`sim/control/attitude/__init__.py`, and `sim/mission/__init__.py` may include
additional workbench or compatibility items beyond the product-facing inventory.

When adding a new controller or mission behavior:

1. Follow [Controller Naming Conventions](project/controller_naming_conventions.md).
2. Check the [Built-In Reference GNC Library](reference-gnc.md) and this
   inventory to decide whether the behavior is product-facing and cataloged.
3. Add or update at least one runnable YAML example when the behavior is
   product-facing.
4. Update this inventory if users should know the feature ships with the
   product.
5. Add focused tests for any promoted user-facing behavior.
