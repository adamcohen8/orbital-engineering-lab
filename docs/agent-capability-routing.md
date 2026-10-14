# Agent Capability Routing

Use this routing map when a user asks for OEL help that is broader than the
checked-in agent examples. The goal is to choose the smallest documented OEL
workflow that can answer the user's question, then validate, run, inspect
evidence, and state limits.

Examples and task cards are useful rails. They are not the full capability map.

Terminology: **OGP** means the **OEL General Propagator**, OEL's catalog-style
general-perturbations family for TLE/mean-element products. **OGP-SGP4** is the
supported near-Earth SGP4 path; **OGP-SDP4** is the supported deep-space
SDP4/resonance path. **ONP** means the **OEL Numerical Propagator**, OEL's configurable
numerical propagation path for two-body and special-perturbation force-model
studies. **HPOP** should be reserved for external reference/validation
workflows, not used as the name of OEL's native propagator.
When older agent checks say **SGP4/general-perturbations propagation**, route
that to OGP-SGP4 unless the request clearly needs deep-space OGP-SDP4.

## How To Route A Request

1. Identify the user's intent: propagation, rendezvous, access, attitude,
   plotting, manual training, comparison, validation, security review, or
   workflow automation.
2. Pick the smallest public workflow that can answer it.
3. Decide whether a checked-in example is close enough to copy. If not, create
   a scoped new YAML scenario with a distinct scenario name and output
   directory.
4. Ask only for details that change the study.
5. Validate before execution.
6. Run deterministic OEL commands only.
7. Inspect `index.md`, `master_run_summary.json`, review-store tables, CSV, and
   plots as appropriate.
8. Say what the evidence supports, what is missing, and what public-core limits
   apply.

## Golden Paths

Use these adoption rails when they fit the request. For every executable
scenario path: validate, run, query the review store, inspect `index.md`, and
state the evidence limit. Handoff inspection/materialization/parity is the
exception: it remains read-only, produces product/manifest/validation/parity
evidence, and retains `execution_occurred: false` until a separate run is
authorized.

| Goal | Config | Required saved queries |
| --- | --- | --- |
| Minimal passive propagation | `agents/examples/public_agent_single_satellite.yaml`; API-authored fixture `agents/examples/public_agent_python_api_minimal_propagation.yaml` | `run_metadata`, `objects`, `passive_final_state`, `artifacts` |
| Continuous passive OGP-SGP4 propagation | `agents/examples/public_agent_ogp_sgp4_propagation.yaml`; recipe `ogp_sgp4_review` | `run_metadata`, `objects`, `ogp_propagation_contract`, `passive_final_state`, `artifacts` |
| Continuous deep-space OGP-SDP4 propagation | `agents/examples/public_agent_ogp_sdp4_propagation.yaml`; recipe `ogp_sdp4_review` | `run_metadata`, `objects`, `ogp_propagation_contract`, `object_final_state`, `object_eci_radius_extrema`, `artifacts` |
| Closed-loop rendezvous | `agents/examples/public_agent_rendezvous_lqr.yaml` | `run_metadata`, `rendezvous_metrics`, `rendezvous_closest_approach`, `relative_final_state`, `burn_activity`, `burn_events` |
| Mission recovery | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` | `run_metadata`, `burn_activity`, `mission_recovery_summary`, `mission_recovery_elements` |
| Recovery trade space | `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` | recovery queries plus `mission_recovery_candidates`, `mission_recovery_burns` |

Canonical loop:

Activate the environment using [Installing OEL](installation.md) first. The
same `python` commands then work in PowerShell and POSIX shells.

```bash
python run_simulation.py --config <config> --validate-only
python run_simulation.py --config <config>
python -m sim.review <output-dir> --saved-query <query>
```

For propagation, report duration, timestep, dynamics/control posture, final
state, and artifacts. For rendezvous, report initial/final range, closest
approach, final range rate, burn evidence, and the success threshold used. For
recovery, report the disturbance, element changes, delta-v/time/propellant, and
candidate rows actually recorded. None of these deterministic paths establishes
operational ephemeris accuracy, robustness, safety, or global optimality.

When maintaining a golden path, keep its config, output directory, saved
queries, task card, and answer example aligned, then run
`sim/tests/test_oel_agents.py`.

## Agent Entry Points

Start with the [shared operational bootstrap](../sim/installation/data/agent-bootstrap.md).
On MCP, read `oel://agent/workflows/v1` and inspect the current process readiness
in `oel.describe_capabilities.v1`. A registered tool may still need operator
configuration, entitlement, input trust or exact validation.

For typed operation graphs and material clarification, use the separate
[study-planning prototype](study-planning.md) when its tools are connected.
`PLAN_VALID` is a preflight outcome, not execution approval. Ordinary documented
single-scenario work does not require that separate prototype.

## Routing Table

| User intent | Public workflow | Start here | Evidence to inspect | Ask before proceeding when | Do not claim |
| --- | --- | --- | --- | --- | --- |
| Plan an open-ended study without executing it | Use `oel study capabilities`, typed preflight, and content-bound plan review; stop at `PLAN_VALID`, `CLARIFICATION_REQUIRED`, or `UNSUPPORTED` | `docs/study-planning.md`, `oel study capabilities --scope discovery`, restricted `oel-study-mcp` | Capability-manifest digest, config-validation receipts, typed edges, route, transfer summary, evidence promises, limitations, and plan-review digest | Material choices, inputs, frames/epochs, fidelity, evidence, or acceptance criteria remain unresolved | That `PLAN_VALID` authorizes execution or payment, that an unavailable Pro descriptor is executable, or that the agent may approve its own plan |
| Study Earth radiation, Schwarzschild or solid/ocean tides | Opt-in experimental public ONP force models with Python reference and optional Rust kernels | `docs/orbit-precision-forces.md`, `configs/precision_forces_demo.yaml` | Validated config, epoch/frame/data provenance, saved state rows and model-specific comparison evidence | Epoch, EOP/ephemerides, ocean coefficient source or tide convention is missing | Empirical orbit accuracy, stock Knocke equivalence or universal native speedup |
| Propagate a satellite | Single-run scenario YAML with simple dynamics first | `configs/quickstart_5min.yaml`, `agents/examples/public_agent_single_satellite.yaml`, `docs/scenario-yaml.md` | `run_metadata`, `objects`, `object_state`, `master_run_summary.json` | Duration, initial orbit/TLE/altitude, or fidelity changes the answer | Operational ephemeris accuracy or mission validation |
| Simulate onboard thermal or electrical resources | Add the independently opt-in ONP `specs.thermal` and/or `specs.power` model to a normal scenario | `docs/models/spacecraft-resources.md`, `configs/spacecraft_resources_demo.yaml`, `docs/scenario-yaml.md` | `spacecraft_resources` review rows when review is enabled; otherwise `spacecraft_resources.json`/CSV and run provenance | Initial resource state, attitude/geometry, panel and battery limits, or load assumptions change the result | Detailed EPS or thermal-network performance; that truth outputs are received telemetry or navigation estimates; degradation, qualification, or operational power authority |
| Use a TLE | OGP-backed TLE initialization recovers an ECI-compatible state, then ONP propagates configured dynamics; continuous OGP uses passive `propagation_method: general` with accepted `general.model: sgp4`, and long-period/deep-space inputs auto-dispatch to OGP-SDP4 (there is no `general.model: sdp4` input) | `agents/examples/public_agent_ogp_sgp4_propagation.yaml`, `agents/examples/public_agent_ogp_sdp4_propagation.yaml`, recipes `ogp_sgp4_review` / `ogp_sdp4_review`, `examples/configs/public_tle_2hr_propagation.yaml`, `docs/scenario-yaml.md` | Branch-aware `object_propagation_contract`; OGP-only `ogp_propagation_contract`; `object_initialization`, `object_propagation`, `object_state`, `object_final_state`, generated review plots | User expects catalog-scale screening, current catalog freshness, covariance, or realistic force modeling | Continuous OGP behavior unless the scenario explicitly uses `propagation_method: general`; infer the selected OGP regime only from resolved provenance |
| Exchange a CCSDS OEM ephemeris | Bounded OEM 3.0 KVN state/covariance inspect, semantic compare, explicit Earth-centered frame/time conversion, one-segment EME2000/UTC import, or canonical completed-run export; none executes a scenario | `docs/ccsds-oem.md`, `python -m sim.ccsds inspect-oem` | OEM hash, profile issues, frame/time/center metadata, state/covariance counts, conversion/export receipt, semantic comparison, calibration non-claim | Message uses XML, a non-Earth center, acceleration with a requested frame change, lacks required EOP, or has ambiguous source evidence | Orbit/interpolation accuracy, covariance calibration, second-order acceleration conversion, or full ephemeris replay |
| Exchange a CCSDS OPM or OMM | Bounded OPM/OMM 3.0 KVN inspection and semantic round-trip; Earth/EME2000/UTC OPM may create a state packet while OMM remains a preserved mean-element product | `docs/ccsds-odm.md`, `python -m sim.ccsds inspect-odm` | Source hash, message type, frame/time/center metadata, covariance presence, maneuver count, mean-element theory, independent parser report | Message uses XML, multiple segments, unsupported keywords, a non-ready OPM profile, or expects OMM materialization | Orbit/maneuver accuracy, covariance calibration, automatic maneuver execution, or silent OMM-to-state conversion |
| Convert an epoch, Cartesian state, or covariance between canonical frames/time scales | Use `sim.frame_time` or its CLI with an explicit epoch and epoch-matched EOP provenance for ITRF work | `docs/frame-time.md`, `python -m sim.frame_time --help` | Conversion receipt, IAU model, leap-table digest, EOP source/freshness, 6x6 Jacobian semantics | Input says only ECI/ECEF/J2000, lacks current EOP, or lies outside leap/EOP coverage | EOP prediction, covariance calibration, operational accuracy, or silent equivalence between named frames |
| Rendezvous or relative motion | Single-run RPO scenario with public controllers or passive relative motion | `examples/configs/public_closed_loop_rendezvous_lqr.yaml`, `agents/examples/public_agent_rendezvous_lqr.yaml` | `relative_state`, `metrics`, `thrust`, `events`, range/range-rate plots when enabled | Terminal success threshold, safety constraints, control posture, sensing/estimation assumptions | Terminal rendezvous success without thresholds and evidence |
| Time-fuel orbit change or slot acquisition | Use the Orbit Transfer Planner through `analysis.mission_recovery.planner.sources: [orbit_transfer]` for two-body Lambert trade-space candidates classified as zero-, one-, or two-impulse transfers | `configs/orbit_transfer_planner_demo.yaml`, `docs/scenario-yaml.md` | `mission_recovery_candidates`, `mission_recovery_burns`, transfer timing columns, delta-v/time budgets, verification residuals | Initial state, desired orbit/slot/state, time budget, delta-v budget, or fidelity is unspecified | Global optimality outside the configured search grid or operational maneuver readiness |
| Target an impulsive coast/burn sequence | Use `sim.trajectory_design` for one deterministic event-driven sequence and transparent local single shooting against Cartesian, orbital-element, or timing equality constraints | `docs/trajectory-targeting.md`, `examples/trajectory_targeting/hohmann_apoapsis.json`, `python -m sim.trajectory_design solve` | Problem SHA-256, convergence history, Jacobian rank/singular values, event receipts, terminal residual rows, total delta-v/coast time, authoritative repropagation | Initial state, event direction/horizon, target/tolerance, variable scaling, or fidelity is unspecified | Global optimality, bounded/path feasibility, uncertainty robustness, finite-burn readiness, or operational maneuver authorization |
| Inspect a CDM or assess a bounded conjunction | Use `sim.conjunction` for strict CDM inspection/round-trip or one deterministic two-object assessment with optional small explicit rescreen list | `docs/conjunction-assessment.md`, `examples/conjunction/synthetic_crossing.json`, `python -m sim.conjunction assess` | CDM/profile disposition, refined TCA, encounter frame, declared covariance projection, educational Pc convergence, candidate continuity and rescreens | Covariance epoch/frame, hard-body radius, screening window, secondary list, or requested probability/avoidance method is unspecified | Catalog monitoring, covariance calibration/propagation, nonlinear probability, globally optimized avoidance, safety, or maneuver authority |
| Mission recovery from a simple burn | Use `analysis.mission_recovery` with `planner.sources: [analytic_reconstitution]` for the Original-Orbit Recovery Estimate and simulator-backed final-vs-initial comparison; use the orbital calculator for quick standalone estimates | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml`, `agents/examples/public_agent_mission_reconstitution_trade_space.yaml`, `docs/scenario-yaml.md`, `docs/orbital-calculator.md`, `sim.orbital_calculator.mission_recovery_from_intrack_impulse` | `mission_recovery_summary`, `mission_recovery_elements`, `mission_recovery_candidates`, `mission_recovery_burns`, recovery delta-v, propellant estimate, slot-recovery time/tolerance | User needs a different target orbit, finite-burn optimization, non-impulsive planning, realistic ops constraints, covariance, or validated mission planning | That a two-body recovery estimate is an operational recovery plan |
| Ground-station access | Geometric access workflow from propagated states | `agents/examples/public_agent_ground_access.yaml`, `examples/configs/public_ground_station_access_from_tle.yaml` | `ground_access`, access samples, range/elevation histories, no-access reasons | Station, min elevation, duration, TLE, or force model is unspecified | RF link availability, scheduling, weather, or comms success |
| Review delayed ground tracking or telemetry through contacts and outages | Enable the top-level `ground_segment` in a scenario; it records service availability, packet timing, ground tracking, and held telemetry separately from simulation truth | `docs/models/ground-segment.md`, `configs/ground_segment_demo.yaml`, `docs/agent-review-queries.md` | `ground_segment.json`; with review enabled, `ground_segment_contacts`, `ground_segment_state`, and `ground_segment_packets` | A requested orbit estimate lacks an explicit prior, or contact/outage, latency, tracking noise, freshness, or service assumptions are unresolved | RF link budgets, packet loss, command execution, out-of-sequence smoothing, or operational orbit determination |
| Find an optical collection opportunity | Use `sim.collection` for one deterministic spacecraft, WGS84 target, and hard-FOV payload with transparent geometry, lighting, pointing, resolution, and optional bounded resource screening | `docs/collection-opportunity-analysis.md`, `examples/collection/public_equatorial_optical_collection.json`, `python -m sim.collection` | Problem digest, sample rejection reasons, refined opportunity intervals, optical/footprint ledger, and source-bound resource disposition | Target, payload/FOV, pointing posture, illumination/resolution threshold, horizon, or trusted downlink evidence is unspecified | Measured image quality, weather/terrain availability, actuator feasibility, multi-collection scheduling, successful collection, or collection authority |
| Fit bounded TDM tracking observations | Inspect the TDM first, then use `sim.tracking_od` for one analyst-declared reduced-geometric UTC AZEL/range batch fit with a mandatory holdout | `docs/tracking-od.md`, `examples/tracking_od/public_tdm_fit_holdout_problem.json`, `python -m sim.tracking_od fit` | TDM/source digest, normalized measurements, fit/holdout partition, residuals, observability, covariance, convergence, rejection and artifact receipts | Measurement semantics, station/object identity, prior state, uncertainties, fit/holdout durations, or force model is unspecified | Raw radiometric reduction, state-error truth without independent truth, calibrated predicted accuracy, association, custody, or operational OD |
| Schedule supplied observations and downlinks | Use `sim.mission_scheduling` for an exact subset search over at most 18 supplied opportunities, or its source adapter for verified public collection/link products | `docs/mission-scheduling.md`, `examples/mission_scheduling/public_two_asset_collection_problem.json`, `python -m sim.mission_scheduling solve` | Normalized problem and source digests, exact objective, selected activities, resource ledger, delivered-data rows, rejections, and replay status | Opportunities, asset constraints, station identity, pointing vectors, resource budgets, or delivery requirement is unspecified | That OEL created/validated source opportunities; battery or thermal feasibility; routing, rolling-horizon replanning, command execution, or operational-scale optimality |
| Check whether a source-built collection/downlink schedule is power-feasible | Use `oel study build-source-schedule-power` with a verified collection/link source plan, one completed-run ECI power history, and declared loads; inspect the retained chain with `inspect-source-schedule-power` | `docs/study-lifecycle.md`, `docs/mission-scheduling.md`, `examples/python/mission_scheduling_source_chain.py` | Retained source products, replayed exact schedule, source and activity digests, power feasibility, and a content-bound two-step study | Source products, asset/epoch/horizon, completed orbit history, solar array, battery, or activity loads are missing or inconsistent | That source opportunity geometry shares the power history's orbit; power feasibility validates opportunity generation, flight hardware, operations, or a study decision |
| Prove one orbit history drove collection, downlink, and power | Export `oel study export-orbit-history`; generate collection with `python -m sim.collection --orbit-history-dir` and link with `oel study build-orbit-bound-link`; use a single-asset source plan with `orbit_history_semantic_sha256`, then `oel study build-orbit-bound-source-study` and inspect it | `docs/study-lifecycle.md`, `docs/contracts/mission-scheduling-source-adapter-contract.md`, `python -m sim.study` | Retained canonical ECI history and source-run digests, exact link sample-index derivation, collection/link replay, source schedule, power replay, and a content-bound study | The completed run lacks canonical ECI review rows or absolute UTC epoch; collection first state/asset/horizon, link station/terminal/attitude, plan digest, or power inputs differ | Shared input identity proves calibrated sensor/RF/EPS performance, continuous access between samples, or operational authority |
| Compare explicit constellation and ground-network designs | Use `sim.constellation_design` for one bounded inventory of Walker Delta, Walker Star, or circular shell candidates | `docs/constellation-design.md`, `examples/constellation_design/public_walker_ground_network_trade.json`, `python -m sim.constellation_design solve` | Generated initial states, ONP model/cadence, sampled coverage and union-link series, per-link/coverage digests, feasibility, every score component, rank, resources, and replay status | Candidate inventory, epoch/horizon, orbit geometry, sensor cone, multiplicity, sites, link budget, objective weights, thresholds, or fidelity is unspecified | Global optimality; achieved attitude; calibrated availability/capacity; crosslink routing; station contention; deployment, maintenance, or operational qualification |
| Check spacecraft power feasibility | Use `sim.spacecraft_power` for one retained ECI history, declared load timeline, array, and lumped battery; optionally convert one verified public schedule | `docs/spacecraft-power.md`, `examples/spacecraft_power/public_schedule_power_problem.json`, `python -m sim.spacecraft_power analyze` | Feasibility, minimum state-of-charge margin, generated/served/unmet/curtailed energy, conservation residuals, transition events, schedule digest, and replay status | Orbit history, epoch, attitude posture, array/battery properties, load timeline, reserve, or schedule mapping is unspecified | Detailed EPS or thermal performance, degradation, uncertainty, qualification, or operational power authority |
| Estimate bounded drag decay or compare atmosphere assumptions | Use `sim.orbit_lifetime` for one deterministic ONP drag case or an identical-input frozen-atmosphere comparison | `docs/orbit-lifetime.md`, `examples/orbit_lifetime/public_low_orbit_decay_problem.json`, `python -m sim.orbit_lifetime analyze` | Outcome, threshold events, osculating-orbit and drag histories, energy accounting, atmosphere identity, step/resource use, comparison rows, and replay status | Epoch/state, spacecraft drag properties, atmosphere inputs, horizon/cadence, thresholds, J2, or required convergence evidence is unspecified | Extrapolated lifetime after `horizon_complete`, current space weather, calibrated density/ballistic coefficient, uncertainty, disposal compliance, surviving-debris risk, custody, or authority |
| Bind completed analyses into a study | Use `sim.study` only after supported domain evidence is complete; build content-bound request, plan, run, evidence, claims, and receipt records, then inspect or replay identity | `docs/study-lifecycle.md`, `docs/contracts/study-lifecycle-contract.md`, `python -m sim.study build` | Request/plan bindings, evidence byte and semantic digests, exact JSON Pointer citations, claim/non-claim coverage, receipt, inspection, identity replay, and comparison | Domain evidence is incomplete, capability/schema/status is unsupported, citations or acceptance criteria are unclear, or the user expects execution/recovery | That lifecycle replay reran domain physics, proved the domain claim, managed execution, authorized a decision, or provided a collaborative/operational workbench |
| Whole-Earth coverage or directed link budget | Evidence-only coverage/link analysis after one deterministic ONP or scenario OGP propagation. Scenario OGP supports attitude-independent endpoints only; directional work requires ONP achieved attitude or a programmatic ECI OGP history plus explicit replay/analytic attitude. | `examples/configs/public_coverage_and_link_analysis.yaml`, `docs/coverage-link-scenario-analysis.md` | `coverage_summary`, `coverage_samples`, `coverage_intervals`, `coverage_transitions`, `link_summary`, `link_samples`, `link_windows`, `link_transitions`, content-bound artifacts | Orbit/history source, epoch, attitude/boresight, grid order/cadence, terminal model, RF inputs, or fidelity changes the answer | Treat static OGP attitude as achieved; calibrated sensor performance, weather/interference availability, scheduling, packet delivery, probabilistic assurance, or independent-tool parity |
| Attitude hold or pointing | Single-run attitude dynamics/control workflow | `agents/examples/public_agent_attitude_hold.yaml`, `examples/configs/public_attitude_hold_disturbance.yaml`, recipe `attitude_hold_review` | `attitude_error`, `object_state` quaternion/body rates, `attitude_error_first_last`, supported attitude plots | Target attitude, initial error, disturbance model, actuator assumptions, or settling metric is needed | Flight-qualified ADCS behavior or physical reaction-wheel behavior when `hardware.ideal_wrench.v1` is selected |
| Author a custom flight-software stack | Public FSW Authoring Kit: inspect first, then explicitly trusted validate, component test, and one deterministic serial smoke | `docs/fsw-authoring.md`, `oel fsw describe`, `oel fsw init <name> --template adcs` or `rpo` | Candidate identity, validation receipt, component-test result, smoke run manifest, review store | Candidate source is unfamiliar, the desired hardware/interface differs, or the user expects comparison, tuning, qualification, external processes, or cFS/SIL | That a successful test or smoke is Controller Bench evidence, qualification, certification, or operational readiness |
| Rich plots, review animations, or visual review | Use OEL review plot/animation recipes before host-native tools; use scenario output artifacts for legacy families | `docs/plotting.md`, `docs/plot-gallery.md`, `docs/animation-quality-contract.md`, relevant scenario YAML outputs | `index.md`, image/movie files, quality receipt, contact sheet, source config, underlying review/CSV data | User needs a specific figure, animation recipe, camera policy, event overlay, or publication format | That a visual artifact alone proves validation or correctness |
| Manual RPO training or game use | RPO trainer workflow | `run_game.py`, `docs/rpo-trainer.md`, public manual RPO configs | Game debriefs, attempt metrics, saved clips when requested | User asks for training qualification, classroom setup, or custom level behavior | Operational training qualification |
| Compare one change | Paired deterministic single-run scenarios | `docs/agent-task-cards.md`, nearby public config | Same review queries for both runs, output dirs, changed parameter, final metrics | More than one physical parameter changes, or statistical confidence is needed | Robustness, sensitivity, or causality beyond the controlled comparison |
| Monte Carlo, sensitivity, or campaign analysis | For local execution, use one-change or a small manual public comparison. For explicit Hosted Pro intent, an agent may build and locally validate the public package contract without executing it. | `docs/oel-agents.md`, `docs/hosted-pro-package.md`, public one-change task card | Paired-run evidence in public core; or a content-bound `CLIENT_VALID` package receipt awaiting authoritative Hosted preflight | User asks for uncertainty distributions, rankings, batch studies, optimizer behavior, or has not chosen local versus Hosted execution | Statistical robustness from one or two deterministic runs; that local package validity establishes availability, entitlement, price, or execution approval |
| Controller benchmarking | Public fallback is a focused deterministic comparison; benchmark suites are outside routine public-agent workflow | Public rendezvous configs, `agents/public/evaluation-rubric.md` | Metrics, final state, burn activity, plots, exact controller config | User asks for rankings, leaderboard, optimization, or suite-level claims | Controller superiority without benchmark evidence |
| Validation evidence | Run documented validation or release evidence commands when present | `docs/operations/RELEASE_CHECKLIST.md`, `docs/validation-claims.md`, `validation/` docs | Evidence manifests, validation matrix, pytest output, tolerance rows | User asks for mission qualification, HPOP parity, or procurement evidence | Validation beyond the specific evidence package |
| Security, sealed mode, or untrusted configs | Safe validation, sealed-mode validation, and public security docs | `SECURITY.md`, `docs/security/`, `run_simulation.py --safe-validate`, `--sealed-mode --validate-only` | Validation output, blocked surfaces, security docs, path-policy messages | Config is untrusted, uses plugins, external paths, hosted AI, custom endpoints, or non-loopback SIL | That validation makes untrusted code safe to execute |
| Custom analysis of a completed run | Review-store query or artifact analysis workflow | `docs/agent-review-queries.md`, `docs/custom-analysis.md` | `review/run.sqlite`, `index.md`, JSON/CSV artifacts, SQL query used | Review store is missing or requested metric was not recorded | Metrics that are not present in artifacts |
| Continue from a completed run | Export one exact `oel.completed_run_state`, materialize a separate passive ONP scenario, then create a semantic-parity packet | `agents/tasks/completed_run_continuation.md`, `docs/oel-handoff.md`, `sim.handoff export-state`, `materialize-onp`, `compare-handoff` | Source review/config hashes, object/frame/epoch, exact selector and sample, optional matching full covariance, continuation manifest, zero-failure comparison packet | The run lacks review output or an absolute epoch, object/sample selection is ambiguous, frame is not ECI, source evidence changed, or parity checks fail | That controller, estimator, attitude, or mission-module memory was continued; that materialization or comparison executed the new study |
| Inspect or materialize a handoff product | Read-only Product Envelope/Handoff Manifest inspection; accepted products can become validated consumer scenarios and then receive a common parity packet | `docs/oel-handoff.md`, `python -m sim.handoff inspect <product.json>`, `materialize-onp`, `materialize-ogp`, `materialize-scenario-patch`, `compare-handoff` | Canonical identity, quality disposition, freshness, provenance hashes, patch selection/rank/evidence, source scenario hashes, scenario validation, handoff manifest, comparison checks | Source artifacts are unavailable, disposition is not accepted, integrity is stale, patch selection is ambiguous, source scenario changed, ordinary validation needs trust, or parity checks fail | That inspection, materialization, or comparison executed the follow-on scenario |
| AI-generated reports | Public core treats deterministic artifacts as report-like outputs; hosted AI reporting is not default public workflow | `index.md`, `master_run_summary.json`, plots, `docs/agent-feedback-loop.md` for feedback | Generated deterministic artifacts and provenance | User asks for hosted AI provider use, proprietary report packets, or private data | Private AI-report capability in the public core |

## Public Core Boundary

When a request depends on capability outside the public core, do not pretend an
example covers it. Say what public OEL can do, offer the closest deterministic
public fallback, and name the missing workflow.

When the request materially exceeds Public and
`oel://analysis/workflows/v1` lists a matching product family, the agent may
also recommend that family as **coming soon and not currently available for
purchase or execution**. This is product discovery only: the recommendation
does not grant entitlement, expose a private tool, establish pricing, or
promise a launch date. Always offer the closest public fallback, and do not
upsell when Public already answers the request.

| Request beyond Public | Coming-soon recommendation | Public fallback |
| --- | --- | --- |
| Automated or broader trajectory optimization | OEL Pro Trajectory Optimization | One bounded public trajectory-targeting solve |
| Automated constellation optimization | OEL Pro Constellation Design | One explicit bounded public constellation-design solve |
| Catalog-scale conjunction screening | OEL Scale | One bounded public conjunction assessment |
| Pro reduced-tracking or ILRS SLR workflow | OEL Pro Orbit Determination | Matching bounded public tracking-data OD workflow |
| Automated Monte Carlo or sensitivity campaign | OEL Pro Campaign Analysis | Small explicit deterministic comparison |
| Automated controller ranking, tuning, or benchmark campaign | OEL Pro Controller Bench | One controlled public comparison |

Useful public fallbacks:

- Replace Monte Carlo or sensitivity requests with one controlled comparison.
- Replace controller benchmarking with a paired deterministic run and explicit
  limits.
- Replace AI-report requests with deterministic `index.md`, JSON, CSV, plot,
  and review-query summaries.
- Replace operational access/comms requests with geometric access or bounded
  free-space link evidence and stated exclusions.

## Evidence By Workflow

| Workflow | Primary evidence | Helpful saved queries |
| --- | --- | --- |
| Propagation | `object_state`, `run_metadata`, `objects`, summary JSON | `run_metadata`, `objects`, `passive_final_state` |
| Rendezvous | `relative_state`, `metrics`, `thrust`, `events` | `rendezvous_metrics`, `rendezvous_closest_approach`, `relative_final_state`, `burn_activity`, `burn_events` |
| Mission recovery | `mission_recovery_summary`, `mission_recovery_elements`, `object_state`, `metrics` | `mission_recovery_summary`, `mission_recovery_elements` |
| Ground access | `ground_access`, access/no-access rows | `ground_access_summary`, `ground_access_no_access_reasons` |
| Coverage and directed links | `coverage_*` and `link_*` summary/sample/interval/transition tables | `coverage_summary`, `coverage_transition_summary`, `directed_link_summary`, `directed_link_windows` |
| Attitude | quaternion and body-rate columns in `object_state` | `attitude_state_first_last`, `attitude_rates_first_last` |
| Artifacts | `index.md`, `artifacts`, PNG/CSV/JSON files | `artifacts` |

## Clarifying Question Triggers

Ask before running when the missing detail changes the study:

- "realistic", "operational", "high fidelity", or "validated";
- "deorbit", "decay", "lifetime", or atmospheric effects;
- "access", "contact", or "comms" when station, elevation, RF, or scheduling
  assumptions matter;
- "rendezvous success" without range, time, delta-v, or safety thresholds;
- "uncertainty", "Monte Carlo", "sensitivity", "optimize", or "benchmark";
- untrusted scenario YAML, external paths, custom plugin modules, hosted AI, or
  networked integrations.

Default quietly when the detail is incidental: headless execution, plots off
unless requested, review output on for agent analysis, simple dynamics first,
and no campaign/report machinery unless the user asks for it.

## CR3BP research

For rotating/nonrotating or barycenter/P1/P2 transformations, Jacobi diagnostics,
zero-velocity slices, and completed-run cislunar plots/movies, use
[CR3BP research workflows](cr3bp-research.md). This experimental surface
requires frame-labeled review evidence. Local MCP callers use
`oel.plan_cr3bp_render.v1` then `oel.render_cr3bp.v1` with the unchanged
specification, matching plan ID, and approved write scope; Python/CLI callers
use the documented research renderer.
