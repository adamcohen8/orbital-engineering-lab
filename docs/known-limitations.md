# Known Limitations

Orbital Engineering Lab is a public beta simulation core. It is useful for
research, education, prototyping, and pre-flight engineering analysis, but it is
not flight-qualified or operational decision-grade software.

## Public Beta Scope

- The public core is optimized for deterministic single-run simulation.
- The public core includes a bounded two-body Lambert solver and grid-based
  orbit-transfer trade-space example. General optimization, gain tuning, Monte
  Carlo, sensitivity studies, controller benchmarking, campaign dashboards,
  AI-assisted reporting, and flight-software workbench/orchestration workflows
  are Pro surfaces and are not included in the public export. Public users may
  still provide an importable custom class implementing the complete
  `SatelliteFlightSoftware` contract; public core does not provide private
  composition, benchmarking, qualification, or promotion tooling for it.
- The primary public surfaces are the CLI, scenario YAML, Python API, review
  query CLI/API, custom review plotting, and RPO trainer.
- Payload and artifact shapes are documented, but some non-contract fields may
  evolve while the project is pre-1.0.
- Public conjunction assessment is limited to one deterministic primary-
  secondary pair, a small explicitly declared rescreen list, covariance
  declared at TCA, and one educational linear 2D Gaussian Pc method. It is not
  catalog screening, covariance propagation, nonlinear probability analysis,
  globally optimized avoidance, or operational maneuver authority; see
  [Conjunction Assessment](conjunction-assessment.md).
- Public integrated studies currently bind completed trajectory-targeting,
  conjunction-assessment, mission-scheduling, orbit-lifetime, and
  spacecraft-power JSON evidence, with at most 12 steps and 16 MiB per retained
  evidence file. Lifecycle replay verifies
  record, citation, and evidence identity; it does not rerun domain physics.
  There is no queue, monitoring, cancellation/resume, migration, viewer,
  campaign, team-signoff, retention, or operational-authorization workflow;
  see [Integrated Study Lifecycle](study-lifecycle.md).

## Validation Status

- The public repo includes unit and regression tests plus curated public
  scenarios.
- The engine has contract docs for scenario YAML, payload artifacts, and
  single-run behavior.
- The [Physics Model Reference](physics-models.md) documents model equations,
  assumptions, config knobs, implementation locations, evidence hooks, and
  limits so validation evidence can be interpreted in context.
- Public-safe external-reference validation evidence should be bundled when it
  is redistributable and tied to a specific claim. Proprietary reference data,
  customer-specific evidence, and large automated validation workflows remain
  private/product surfaces.
- Users should independently validate behavior for their mission envelope,
  force models, time spans, controller assumptions, and numerical tolerances.
- Public scenarios such as `configs/ric_pd_10km_experiment.yaml` are review
  workflows and examples. They are not mission qualification evidence by
  themselves; see [Validation Claims](validation-claims.md).

## Scenario Safety

Only run scenario YAML files from sources you trust. Scenario configs can point
at importable Python modules/classes for controllers, guidance, mission
strategies, and mission execution modules. That extension model is powerful, but
loading an untrusted scenario can run untrusted Python code.

## Compatibility

The package declares Python `>=3.10,<3.15`. The blocking local release gate
exercises the default public acceptance path on Python 3.11. Hosted workflows
are manual advisory diagnostics, not public CI gates. The wider release
compatibility program requires retained evidence for claimed Python/operating-system rows;
see [`compatibility.md`](compatibility.md). Installation and recovery commands
for Windows, macOS, and Linux are maintained in
[`installation.md`](installation.md). If you depend on a particular row, verify
that release evidence and run acceptance in the target environment.
Python 3.9 is no longer a supported procurement baseline because several
dependency vulnerability fixes require Python 3.10 or newer.

## Modeling Limits

- **OGP** means the **OEL General Propagator**: OEL's catalog-style
  general-perturbations family for TLE/mean-element products. The current OGP
  implementation supports **OGP-SGP4** for near-Earth SGP4 and **OGP-SDP4**
  for deep-space/resonance TLEs at or above the 225-minute period threshold.
- **ONP** means the **OEL Numerical Propagator**: OEL's configurable numerical
  propagation path for two-body and special-perturbation force-model studies.
  ONP is distinct from passive catalog-style OGP propagation. HPOP names
  external reference/validation workflows, not the native OEL propagator.
- TLE initialization samples OGP into an ECI-compatible initial state.
  Subsequent propagation uses the configured ONP force model; it does not
  continue as catalog-style OGP propagation unless the object explicitly uses
  `propagation_method: general` with `general.model: sgp4`. The legacy
  Keplerian mean-element initializer is available only as an explicit opt-in.
- OGP objects are passive catalog-style objects in the current
  implementation. They do not accept thrust, orbit controllers, maneuvers, or
  covariance. TLEs with orbital period at or above 225 minutes dispatch to
  OGP-SDP4 instead of being treated as near-Earth SGP4. Continuous simulation
  histories are canonical ECI even when product metadata requests native TEME;
  direct OGP and Scale batch interfaces expose native TEME arrays. The engine's
  Vallado IAU-76/FK5 + IAU-80 TEME-to-ECI transform retains legacy defaults
  inside older runtime call sites. Interchange callers should use the bounded
  [`sim.frame_time`](frame-time.md) contract, which supports UTC/TAI/TT,
  sampled UT1, EME2000/TEME/GCRF/ITRF state and 6x6 covariance transforms, and
  fails closed on ambiguous frame names, stale leap-second coverage, or
  missing EOP provenance. GCRF/ITRF uses an IAU 2006/2000A CIO path with
  local finals2000A/C04-style EOP ingestion and freshness audit; OEL does not
  download or predict EOP. Scenario-level Earth-fixed transforms can use
  `simulator.frames.model: iau76_80_eop` with an EOP file and are recorded in
  `frame_provenance`; neither surface is operationally qualified.
- Ground-station access is passive and geometric. It tracks line of sight,
  elevation, and range. The separate public directed-link analysis adds an
  inspectable free-space budget with scalar or hard-cone gain, but neither
  workflow models weather, interference, calibrated equipment, scheduling, or
  command/telemetry behavior.
- Public coverage and link products are sampled deterministic engineering
  analyses. They do not establish calibrated sensor performance, operational
  RF availability, exact swept footprints, probabilistic availability, or
  independent external-tool parity. Runtime monitoring is limited to an
  explicitly authorized directed link; whole-Earth coverage is postprocessed.
- Public optical collection opportunities are limited to one WGS84 surface
  target and one hard-FOV optical payload over one deterministic ONP history.
  The v1 workflow uses simple-GMST Earth rotation, an analytic-enhanced Sun,
  ideal target tracking, first-order diffraction/sampling resolution, local
  tangent-plane footprint metrics, and independent per-opportunity storage and
  supplied-downlink screening. It does not model terrain, weather, clouds,
  refraction, radiometry/MTF, jitter/smear, actuator torque, exact swept area,
  multi-collection resource timelines, radar performance, or calibrated and
  operational availability.
- Public multi-asset mission scheduling is an exact subset search over at most
  18 caller-supplied opportunities. It enforces per-asset non-overlap,
  direct-rate slew/settling, horizon energy, event-based storage, payload duty
  cycle, shared-station contention, and observation delivery. It does not
  validate the source opportunities or model batteries, thermal state,
  packets, crosslinks, routing, uncertainty, disruptions, rolling-horizon
  replanning, command execution, or operational-scale optimization.
  The source adapter can ingest completed public optical-collection and
  directed-link artifacts, but it does not re-run their physics during ordinary
  conversion, infer product freshness beyond content identity and declared
  epoch/horizon, or derive an absolute downlink pointing vector absent from the
  directed-link v0.1 product.
- Public spacecraft-power analysis accepts one retained ECI history and one
  declared load timeline for at most seven days. It uses an analytic Sun,
  cylindrical or conical Earth shadow, ideal Sun tracking or supplied retained
  attitude, one solar array, and one lumped battery. It does not model thermal
  state, temperature effects, degradation, self-shadowing, detailed EPS/bus
  topology, uncertainty, probabilistic availability, hardware qualification,
  or operational authority. Its schedule adapter verifies and converts one
  completed public schedule but does not add battery dynamics to the scheduler.
- Public orbit-lifetime analysis accepts one ECI state and a horizon of at most
  90 days with at most 500,000 ONP RK4 steps. Atmosphere and space-weather
  values are explicit frozen assumptions; OEL does not fetch, predict, or
  calibrate them. Refined events are instantaneous geocentric-altitude
  crossings; endpoint brackets are supplemented by an interior radial-minimum
  check, but consequential studies still require step-convergence evidence.
  Inputs must remain inside the selected atmosphere model's retained altitude
  domain, and non-stopping runs terminate at that domain or the Earth surface.
  `horizon_complete` is not a lifetime extrapolation. The workflow does not
  quantify uncertainty, establish disposal compliance, assess surviving-debris
  risk, maintain orbit custody, qualify software, or authorize operations.
- Opt-in ground-station measurements are synthetic geometric rows. The public
  tracking-OD slice also accepts one bounded CCSDS TDM 2.0 KVN dataset when the
  analyst explicitly declares UTC AZEL and one-way unambiguous range to be
  reduced geometric observables. It provides a single-object batch fit,
  covariance, residuals, and explicit holdout prediction, but not raw
  radiometric reduction, calibrated sensor processing, association, custody,
  calibrated predicted orbit accuracy, or operational tracking authority.
  Governed, calibrated, bulk, and customer workflows remain Pro.
- Spherical-harmonic gravity can use inline terms, user-provided coefficient
  files, or the managed digest-pinned EGM96 source. HPOP/GGM03 reference data
  is not distributed with the public core; ICGEM and HPOP/GGM03 require an
  explicit local path. Managed downloads require `allow_download` and are
  blocked in sealed mode.
- Atmospheric re-entry diagnostics are first-pass aero/thermal estimates. They
  use atmosphere-relative speed, configured drag area/coefficient, and
  Sutton-Graves heat rate. Optional satellite lift is coefficient/vector based
  and intended for first-pass atmospheric steering studies, with vehicle aero
  properties supplied through `objects.<id>.specs.aero`. Reported re-entry
  g-load is aerodynamic drag load, not total acceleration during thrusting.
  The model does not include ablation, plasma, breakup debris, plume heating,
  high-fidelity hypersonic aerodynamics, or detailed thermal protection
  response.
- The bundled RPO trainer is educational and interactive; it is not a
  certification, mission-assurance, or operational training system.
