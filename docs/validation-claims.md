# Validation Claims

This page states v1.0.0 local-release validation claims only where the evidence
names a bounded use and retained source/config identity. A local branch or
unsigned release candidate is not proof that a public GitHub release has been
published. This page is intentionally conservative: the open-source edition
should be useful and credible without implying decision-grade mission
assurance.

## Current Public Claims

The public repository supports these claims:

- Scenario YAML can be parsed, validated, and executed through the documented
  CLI and Python API entrypoints.
- Deterministic single-run two-body, relative-motion, attitude, sensing,
  estimation, and control examples are covered by unit and regression tests.
- Curated public scenarios produce reproducible JSON, CSV, Markdown, and PNG
  artifacts when run in a supported local Python environment.
- The public model-reference docs describe the configured physics equations,
  assumptions, implementation locations, evidence hooks, and limitations needed
  to interpret validation results; see
  [Physics Model Reference](physics-models.md).
- The flagship RIC_PD 10 km scenario demonstrates a closed-loop RPO workflow
  with tuned public controller gains and attitude-gated thrust application under
  the assumptions written in `configs/ric_pd_10km_experiment.yaml`.
- The focused RIC_PD 10 km validation posture documents the flagship scenario's
  intended gates and limitations. The public export can rerun the scenario and
  companion analysis script. Generated release evidence and private
  qualification packets remain outside the source export.
- The public payload and artifact surfaces are documented in the engine,
  scenario YAML, and payload contracts.
- The public [offline validation suite](tracking-od.md#offline-release-validation)
  runs synthetic force and orbit determination checks against an installed
  runtime. Its claims are limited to the tested inputs and tolerances.

For bounded analysis surfaces, claims are limited to the following retained
evidence and their current contract versions. A later change needs its own
focused replay before it inherits a claim:

- frame/time external evidence covers the retained epoch, state, EOP inputs,
  and stated residual envelopes; routine replay recomputes retained Orekit
  residuals against the current implementation;
- OEM and OPM/OMM external cross-reads cover only the metadata, selected state,
  covariance, maneuver-count, and mean-element fields enumerated by their
  retained reports; other supported fields have OEL-only contract/round-trip coverage;
- orbit-lifetime evidence combines analytic checks, deterministic authoritative
  replay, and fixed-state external component diagnostics, but has no independent
  end-to-end propagated-lifetime reference; and
- study-lifecycle evidence validates content identity and citation binding.
  Its `VC-0` through `VC-4` values are an authoring vocabulary, while every v1
  capability is capped at `VC-1`; acceptance within that ceiling is not an
  independent maturity assessment.

These rows are registered in the private validation matrix as non-release-
blocking. The public source release carries rerunnable tests, synthetic
recipes, and redistributable inputs where licensing permits, but not generated
validation evidence. A row remains visibly
`MISSING_MANIFEST`, `FAIL`, `STALE`, or `INCOMPLETE` unless the exact focused
tests and retained replay have produced current content-bound evidence.

## Explicit Non-Claims

The public repository does not claim:

- flight qualification,
- operational decision authority,
- safety certification,
- validated performance for arbitrary mission envelopes,
- high-fidelity force-model accuracy for all orbit regimes,
- equivalence to STK, FreeFlyer, GMAT, Orekit, Basilisk, MATLAB/Simulink, or any
  program-specific truth model,
- correctness of user-supplied plugin modules, controllers, or scenario YAML.

## Evidence Artifacts

For public users, the first evidence path is:

```bash
.venv/bin/python run_simulation.py --quickstart
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
.venv/bin/python examples/python/flagship_analysis.py
```

Review:

- `outputs/quickstart_5min/index.md`
- `outputs/flagship_ric_pd_10km/index.md`
- `outputs/flagship_ric_pd_10km/master_run_summary.json`
- `outputs/flagship_ric_pd_10km/custom_analysis/flagship_metrics.json`

For the focused validation claim and gates, see
[RIC_PD 10 km Validation Package](validation-ric-pd-10km.md).

Selected external comparisons are published as recipes with their commands,
prerequisites, and tolerances. Users can regenerate evidence from the approved
inputs. The development repository retains generated release evidence and
private qualification campaigns; those generated packets are not part of the
public source or wheel.

Model-reference pages are part of that traceability chain. They document what
equations and assumptions a validation result exercised, but they do not replace
tests, external-reference comparisons, or scenario-specific evidence.

## User Responsibility

Users should independently validate:

- initial conditions and scenario assumptions,
- force-model fidelity,
- controller and actuator limits,
- sensor and estimator assumptions,
- numerical tolerances,
- runtime environment and dependency versions,
- mission-specific safety, legal, and compliance requirements.

When in doubt, treat OEL outputs as simulation evidence for review, not as an
authorization to act.
