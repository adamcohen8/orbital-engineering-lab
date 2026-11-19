# Physics Model Reference

This page is the entry point for Orbital Engineering Lab's physics-model
documentation. It explains how model equations, implementation code,
configuration, and validation evidence fit together.

These pages are intentionally conservative. They document the equations and
assumptions OEL uses for simulation review, education, and engineering
prototyping. They do not claim flight qualification, operational decision
authority, or validity outside the configured model envelope.

## Why Model Documents Matter

Validation evidence is only useful when the model being validated is clear. A
completed run can show that a scenario produced repeatable outputs, and an
external-reference comparison can show agreement for a specific case, but a
reviewer still needs to know:

- which equations were active,
- which frames, units, and sign conventions were used,
- which simplifications or perturbations were enabled,
- where the implementation lives,
- which tests or reference workflows exercise that model, and
- which claims are not supported by the evidence.

OEL's validation story therefore has three layers:

1. Model specification: the governing equations, assumptions, configuration
   knobs, and implementation locations.
2. Implementation evidence: unit tests, regression tests, contract tests, and
   scenario validation that show the code behaves as intended.
3. Reference evidence: comparisons against analytic solutions or independent
   tools for selected envelopes, such as HPOP orbit cases, Basilisk attitude
   cases, SGP4 propagation checks, or scenario-specific validation packages.

Model documentation does not replace validation. It makes validation
auditable.

## Model Reference Pages

The detailed pages under `docs/models/` are organized by model family:

| Model family | Start here | Scope |
| --- | --- | --- |
| Orbit dynamics | [Orbit Dynamics](models/orbit-dynamics.md) | Two-body propagation, numerical integration, object state units, and orbit-propagation boundaries. |
| Relative motion | [Relative Motion](models/relative-motion.md) | RIC/Hill frame conventions, relative-state construction, and relative-motion controller assumptions. |
| Attitude dynamics | [Attitude Dynamics](models/attitude-dynamics.md) | Quaternion and body-rate propagation, rigid-body torque response, attitude substepping, and disturbance coupling. |
| Cooperative docking | [Docking Dynamics](models/docking.md) | Paired six-degree-of-freedom contact, compliant capture and joined rigid-body propagation. |
| Environment perturbations | [Environment Perturbations](models/environment-perturbations.md), [Earth precision forces](orbit-precision-forces.md) | Gravity harmonics, atmosphere/drag, SRP, third bodies, albedo/IR, relativity, solid/ocean tides, eclipse and re-entry diagnostics. |
| Actuators | [Actuator Models](models/actuators.md) | Force and torque limits, allocation, propulsion devices, attitude actuators, faults, and applied-command logging. |
| Sensors and measurements | [Sensor And Measurement Models](models/sensors-and-measurements.md) | Synthetic own-state/relative measurements, access, cadence, latency, dropout, composition, and typed event boundaries. |
| Evidence traceability | Private Model Validation Map | How model specs connect to tests, validation suites, reference comparisons, review stores, and explicit non-claims. |

## Traceability Pattern

For model-facing work, reviewers should be able to trace a result through this
chain:

```text
scenario YAML
  -> selected model and configuration
  -> governing equations and assumptions
  -> implementation module
  -> tests or validation workflow
  -> run artifacts and review evidence
  -> stated limitation or claim
```

For example, a TLE-initialized numerical propagation scenario should identify
whether the TLE is only used to initialize state or whether the object uses the
passive OGP-SGP4 general-perturbations path. A rendezvous scenario should identify
whether relative states are represented in rectangular or curvilinear RIC and
whether the controller assumes linearized relative motion. A high-fidelity
orbit validation case should state which gravity, atmosphere, solar-radiation,
and third-body terms were enabled.

## Public And Pro Evidence

The public core includes model documentation, scenario YAML validation, unit and
regression tests, curated examples, review-store inspection, and conservative
validation claims. Public users can inspect and rerun documented examples, but
they should independently validate behavior for their mission envelope.

The private/Pro workspace contains additional validation automation and
external-reference workflows, including HPOP and Basilisk comparisons. Those
workflows are useful engineering evidence for specific tested cases. They still
require interpretation, traceability, and mission-specific qualification before
supporting decision-grade use.

See [Validation Claims](validation-claims.md) and
[Known Limitations](known-limitations.md) for the public validation posture.
Private maintainers should also use the Validation Operations guide.

## Writing Or Updating Model Docs

When adding a new model or changing an existing one, update the relevant model
page together with code, tests, and scenario docs. A good model-doc update
answers:

- What equations or algorithmic rules changed?
- What frames, units, or sign conventions matter?
- Which config fields select the behavior?
- Which source files implement it?
- Which tests or validation suites exercise it?
- What limitations or non-claims should a user see before relying on it?

Prefer narrow, auditable claims over broad credibility language. If a behavior
has not been compared against an external reference, say that plainly and point
to the best available implementation or scenario evidence.
