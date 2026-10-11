# Documentation Index

Use this page as the documentation map. The goal is to give each audience a
short path instead of making every document look equally important.

## New Public User

Start here if you want to install OEL, run a scenario, and understand the
public simulation core.

1. [Install On Windows, macOS, Or Linux](installation.md)
   [Build the public Rust backends from source](native-source-build.md).
2. [Create And Use A Workspace](workspaces.md)
3. [Quickstart](quickstart.md)
   — then try [your first relative-motion study](first-relative-study.md).
4. [Update, Audit, Adopt, Or Roll Back](updating.md)
5. [Compatibility And Install Profiles](compatibility.md)
6. [Examples Matrix](examples-matrix.md)
7. [Scenario YAML](scenario-yaml.md)
8. [Python API](python-api.md)
9. [Known Limitations](known-limitations.md)

## Evaluator Or Buyer

Start here if you are deciding whether OEL is credible for research,
education, engineering prototyping, or Pro workflow evaluation.

1. [Product Inventory](product-inventory.md)
2. [Flagship RIC_PD 10 km Scenario And Validation](validation-ric-pd-10km.md)
3. [Physics Model Reference](physics-models.md)
4. [Validation Claims](validation-claims.md)
5. [Public Core And Pro Boundary](public-vs-pro.md)
6. [Security And Procurement](security/supply-chain.md)
7. [Compatibility And Install Profiles](compatibility.md)

## Agent User

Start here if you want an AI coding assistant to create, validate, run, or
inspect OEL scenarios.

1. [OEL Agents](oel-agents.md)
2. [Agent Capability Routing And Golden Paths](agent-capability-routing.md)
3. [Agent Task Runner](agent-task-runner.md)
4. [Agent Run Lifecycle: Tip And Nap](agent-run-lifecycle.md)
5. [Agent Task Cards](agent-task-cards.md)
6. [Agent Review Query Recipes](agent-review-queries.md)
7. [Agent Feedback Loop](agent-feedback-loop.md)
8. [Agent Evaluation Packet](agent-evaluation-packet.md)
9. [Supported Local MCP Surface](oel-mcp.md)
10. [Local BYO Study Planning Prototype](study-planning.md)
11. [Public Hosted OEL Client](hosted-oel-client.md)
12. [Hosted Execution Package Contract](hosted-pro-package.md)

The root `AGENTS.md` and `agents/public/AGENTS.md` are the agent playbooks.
The docs above explain the supporting workflow, evidence, evaluation, and
feedback material.

## Workflow User

Start here when you already know the kind of work you want to run.

| Goal | Start with |
| --- | --- |
| Edit or author scenario YAML | [Scenario YAML](scenario-yaml.md) |
| Update a managed engine safely | [Updating OEL](updating.md) |
| Migrate a scenario config explicitly | [Scenario Config Migrations](config-migrations.md) |
| Install in an air-gapped environment | [Offline Installation](offline-installation.md) |
| Inspect completed outputs | [Review Store Contract](review-store.md) |
| Analyze whole-Earth coverage or directed links | [Coverage And Link Scenario Analysis](coverage-link-scenario-analysis.md) |
| Make custom tables or plots from a run | [Custom Analysis](custom-analysis.md) |
| Configure built-in figures | [Plotting](plotting.md) |
| Create and QA agent-native review animations | [Animation Quality Contract](animation-quality-contract.md) |
| View maintained plot examples | [Plot Gallery](plot-gallery.md) |
| Understand physics model assumptions | [Physics Model Reference](physics-models.md) |
| Explore actuator models | [Actuators](actuators.md) |
| Select built-in controllers and command modules | [Built-In Reference GNC Library](reference-gnc.md) |
| Author and smoke-test a custom public FSW stack | [Public Flight Software Authoring](fsw-authoring.md) |
| Use the orbital calculator | [Orbital Calculator](orbital-calculator.md) |
| Use game/training mode | [RPO Trainer](rpo-trainer.md) |
| Explore the browser multiplayer Beta | [RPO Duel Beta](../web/rpo-duel-prototype/README.md) |
| Use ML/RL wrappers | [ML/RL Policy Contracts](ml-rl-contracts.md) |
| Connect a local MCP host | [Supported Local MCP Surface](oel-mcp.md) |
| Preflight an open-ended BYO-agent study proposal | [Local BYO Study Planning Prototype](study-planning.md) |
| Prepare and locally validate a Hosted analysis package | [Hosted Execution Package Contract](hosted-pro-package.md) |
| Route an approved proposal to public-local or Hosted OEL | [Public Hosted OEL Client](hosted-oel-client.md) |
| Start an agent run and await durable completion | [Agent Run Lifecycle](agent-run-lifecycle.md) |
| Inspect, import, export, or compare CCSDS OEM ephemerides | [CCSDS OEM Interoperability](ccsds-oem.md) |
| Inspect, import, round-trip, or compare CCSDS OPM/OMM products | [CCSDS OPM And OMM Interoperability](ccsds-odm.md) |
| Convert canonical frames, time scales, states, or covariances | [Canonical Frames And Time](frame-time.md) |
| Build and solve a deterministic event-driven coast/burn target | [Event-Driven Trajectory Targeting](trajectory-targeting.md) |
| Inspect a CDM or assess a bounded two-object conjunction | [Conjunction Assessment](conjunction-assessment.md) |
| Generate a bounded optical collection opportunity | [Optical Collection Opportunity Analysis](collection-opportunity-analysis.md) |
| Fit reduced-geometric CCSDS TDM observations and inspect holdout predictions | [CCSDS TDM Tracking Orbit Determination](tracking-od.md) |
| Schedule supplied observations and downlinks across a small multi-asset fleet | [Bounded Multi-Asset Mission Scheduling](mission-scheduling.md) |
| Compare a bounded set of Walker/shell constellations and ground networks | [Constellation And Ground-Network Design](constellation-design.md) |
| Check eclipse-coupled solar-array and battery feasibility from a retained orbit and load timeline | [Spacecraft Power Analysis](spacecraft-power.md) |
| Simulate one-node thermal and bounded battery state during ONP propagation | [Spacecraft Thermal and Power Resources](models/spacecraft-resources.md) |
| Model delayed ground tracking, telemetry, and service outages | [Headless Ground Segment](models/ground-segment.md) |
| Propagate one deterministic drag-decay case or compare frozen atmosphere assumptions | [Deterministic Orbit Lifetime Analysis](orbit-lifetime.md) |
| Bind completed analyses into one evidence-cited, content-bound study | [Integrated Study Lifecycle](study-lifecycle.md) |

## Pro Workspace User

These docs are available in the full private/Pro workspace and are omitted from
the public export when they describe private workflow acceleration or operating
processes.

1. Pro User Guide
2. Pro Python API
3. Campaign Analysis
4. Controller Bench
5. Pro Covariance Analysis
6. Pro AI Reports
7. Pro AI Config Assistant
8. Pro GNC Workbench
9. Validation Operations
10. Pro Communications Analysis Contract
11. Pro Capability Routing (including communications workflows)

In the Pro workspace, use the repository root README for buyer-facing Pro
positioning and local handoff notes for project continuity.

## Maintainer Or Contributor

Start here when changing behavior, contracts, release posture, or
public/private packaging.

1. [Engine Contract](contracts/engine-contract.md)
2. [Scenario YAML Contract](contracts/scenario-yaml-contract.md)
3. [Payload And Artifact Contract](contracts/payload-artifact-contract.md)
4. [Review Store Contract](review-store.md)
5. [Controller Naming Conventions](project/controller_naming_conventions.md)
6. [Built-In Reference GNC Library](reference-gnc.md)
7. [Data Handling And Boundary Statement](security/data-handling.md)
8. [Security Incident Process](security/incident-response.md)
9. [Installer And Updater Threat Model](security/installer-updater-threat-model.md)

Private maintainers should also use the Validation Operations and Release
Checklist guides available in the full workspace.

Private release/export operating notes live under `docs/operations/` in the
full workspace and are intentionally excluded from the public export.

## Internal Planning

Long-form roadmap, maturity, founder, and commercial-readiness files are
planning memory, not first-run or buyer-facing documentation. Keep them out of
the normal user path unless the task is explicitly about product strategy,
validation investment, release governance, or business/compliance planning.

- [Optional Earth precision forces](orbit-precision-forces.md): public radiation, relativity and tidal gravity with a synthetic-data demonstration.
