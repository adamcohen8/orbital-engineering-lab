# OEL Agent Instructions

For operating OEL on a user's behalf, start with
[`sim/installation/data/agent-bootstrap.md`](sim/installation/data/agent-bootstrap.md).
Managed workspaces receive that same guide as `AGENTS.md`; MCP serves it at
`oel://agent/bootstrap/v1`. Discover active workflow routes with
`oel://agent/workflows/v1` and process readiness with
`oel.describe_capabilities.v1`. The detailed rules below also cover repository
maintenance. For the separate typed study-planning prototype, see
[`docs/study-planning.md`](docs/study-planning.md); do not assume its tools are
available on the main MCP connection.

Orbital Engineering Lab agents should orchestrate documented workflows. They
should not replace, approximate, or silently bypass the deterministic physics
engine.

Agents are general helpers for OEL, not runners for a fixed example catalog.
Use examples and task cards as onboarding rails and regression fixtures, then
generalize from the documented interfaces to the user's actual question.

This file is intentionally public-safe. It helps AI coding agents such as
Codex, Cursor, Claude Code, Gemini CLI, and Grok Build work with the
open-source OEL core.
For the fuller public agent playbook, read `agents/public/AGENTS.md` and
`docs/oel-agents.md`. For routing broad user intents to documented workflows,
read `docs/agent-capability-routing.md`. For repeatable recipe execution,
completed-run inspection, comparison packets, and standard review plots, read
`docs/agent-task-runner.md`.

For code navigation after the God-file decomposition, use the stable façade
first, then follow the public implementation maps in
`docs/config-api-architecture.md`, `docs/runtime-architecture.md`, and
`docs/plotting.md` to the focused owner.

## Instruction Scope And Product Boundary

- Follow the closest applicable `AGENTS.md`. This root file is the public-safe
  baseline, and `agents/public/AGENTS.md` expands public workflows. A private
  workspace may provide closer instructions for its authorized surfaces.
- Keep public and non-public evidence separate. Do not copy private source, configs,
  customer inputs, provider responses, validation references, or generated
  report packets into public files or public answers.
- Public-core single-scenario execution uses deterministic serial object
  stepping. Automatic within-scenario object workers, hierarchical
  campaign/object planning, Monte Carlo, sensitivity, config queues, and
  controller benchmarks are not public-core capabilities. When those are unavailable,
  use the documented deterministic public fallback rather than recreating
  private orchestration.
- Opt-in thread transports for object steps, campaigns, controller studies,
  and Scale remain Pro workflows. They do not change public serial stepping
  or grant entitlement to an unavailable workflow.
- When `oel://analysis/workflows/v1` lists a matching Pro escalation for a
  request that materially exceeds the public workflow, an agent may recommend
  that named product family as
  **coming soon and not currently available for purchase or execution**. It
  must also offer the documented public fallback.
  Recommendation metadata is not entitlement or execution authority: do not
  invent pricing, dates, access, private tools, or an unnecessary upsell when
  Public can answer the request.

## Default Agent Posture

Install and activate OEL through `docs/installation.md` before running project
commands. Commands in this playbook use portable `python`; if activation is
unavailable, substitute the explicit PowerShell or POSIX interpreter path from
that guide.

- Treat scenario YAML, versioned standalone orbital-analysis problems, CLI
  commands, Python APIs, tests, docs, and generated
  artifacts as the supported interface.
- Treat **OGP** as the product name for the **OEL General Propagator**:
  OEL's catalog-style general-perturbations family for TLE/mean-element
  products. **OGP-SGP4** is the supported near-Earth SGP4 path, and
  **OGP-SDP4** is the supported deep-space/resonance SDP4 path.
- Treat **ONP** as the product name for the **OEL Numerical Propagator**:
  OEL's configurable numerical propagation path for two-body and
  special-perturbation force-model studies. Use **OGP** for explicit passive
  catalog-style general-perturbations propagation or SGP4/SDP4-family
  mean-element products. Reserve **HPOP** for external reference/validation
  workflows or legacy command names, not as the name of OEL's native
  propagator.
- Interpret the user's intent first. Route propagation, control, sensing, and
  time-history simulation to scenario YAML. Route a supported standalone
  targeting, conjunction, collection, scheduling, power, lifetime, CCSDS,
  frame/time, OD, or study-lifecycle question to its versioned problem or
  product contract. Choose a nearby example only when it genuinely fits.
- Prefer small, inspectable changes that match existing OEL patterns.
- Start with the simplest deterministic scenario that can answer the user's
  question. Do not add unrequested physics, sensors, estimators, controllers,
  plots, animations, or campaign machinery.
- Generate scenario YAML from natural language only when the resulting config
  can be validated before execution. For user-provided or otherwise unfamiliar
  YAML, run `--safe-validate` before ordinary importing validation.
- Treat `--safe-validate` as an inspection boundary, not permission to execute
  an untrusted config. Trust referenced plugins, modules, and external paths
  before ordinary validation or execution.
- Run `python run_simulation.py --config <path> --validate-only` before running
  a new or edited scenario.
- Treat unknown-field errors as intent failures; do not remove or rename fields
  until the normalized contract is understood. Non-empty Cartesian initial
  states require both position and velocity, and scenarios must select exactly
  one orbital-state form.
- Use the checked-in physics models, controllers, mission logic, and output
  writers. Do not invent shortcut physics in agent scripts or reports.
- Prefer the review store query API over ad hoc parsing of large run logs when
  `review/run.sqlite` is available.
- For a trusted deterministic single-scenario OEL run expected to take about
  30 seconds or longer, prefer the durable lifecycle: use `oel runs start`,
  retain its full accepted handle, perform one bounded `oel runs await`, and
  identity-bind both await and inspect with the accepted manifest reference and
  normalized-config digest before reading simulation evidence. See
  `docs/agent-run-lifecycle.md`. Do not use the lifecycle for arbitrary commands
  or release workflows.
- For open-ended study planning without execution, use the public `oel study`
  capabilities, preflight, and plan-review contracts or the restricted
  `oel-study-mcp` adapter described in `docs/study-planning.md`. Treat
  `PLAN_VALID` as a reviewable proposal only: it is not execution approval,
  payment authorization, entitlement, scientific qualification, or permission
  for an agent to approve its own plan. Public-local plans keep a zero OEL
  execution amount; unavailable Pro descriptors remain non-executable.
- Use `python -m sim.agent_task` when a bundled recipe, comparison,
  standard plot, or portable `agent_evidence_packet.json` would make the
  workflow more reproducible.
- Use `python -m sim.handoff inspect <product-or-manifest.json>` and
  `validate-product` for read-only interchange inspection. A valid product can
  still be blocked from promotion by disposition, freshness, or compatibility;
  inspection never materializes or executes a scenario. Use
  `materialize-onp` only for an accepted, current, compatible state product;
  it writes and validates a passive ONP scenario plus a handoff manifest but
  deliberately does not execute the scenario.
- Use `python -m sim.handoff export-state <completed-run>` to select
  one exact final, sample-indexed, time-indexed, or event-associated ECI review
  row. Require an absolute initial epoch, resolve multi-object runs explicitly,
  and materialize the resulting product as a new study rather than mutating the
  completed run.
- After materialization, use `python -m sim.handoff compare-handoff`
  to verify producer/product, scenario, manifest, and optional first consumer
  review-row semantics. A parity packet is continuity evidence, not physics
  accuracy, and comparison never authorizes or performs execution.
- For bounded offline interchange and object composition, use `oel io formats`,
  `oel io object describe-import`, and `oel io object describe-export` before
  selecting the documented route in `docs/data-io.md`. IO inspection,
  conversion, composition, and export never execute a simulation or rerun OD.
  Preserve explicit object/sample selection, frames, epochs, units, promotion
  gates, and adjacent hash-bound receipts. A trajectory sample seeds a new
  study; complete checkpoints/snapshots have separate continuation contracts.
  Validate every materialized or composed scenario before execution.
- Export external trajectories, states, attitudes, and tables from completed,
  identity-verified review evidence through `oel io export-review`,
  `oel io object export`, or `oel io export-table`. Keep source inputs and
  completed outputs unchanged. Review-to-OGP/TLE fitting is a separate Pro
  action; format conversion cannot turn an osculating state into mean elements.
- Explain orbital mechanics, equations, controllers, and outputs from public
  source and public docs only.
- Before claiming profile support, inspect effective maturity with
  `python -m sim.flight_software list --kind profile --json`; a catalog's declared
  maturity is not current qualification evidence.
- Distinguish stack navigation belief from raw measurements and the separate
  knowledge estimator. See `docs/flight-software-observations.md` for access
  conditions, navigation selection, and timestamped review evidence.
- Enable full-state EKF innovation recovery only when requested or necessary
  for the study. `docs/innovation-recovery.md` describes the off-by-default,
  experimental shared policy for absolute and HCW/SS-J2/TH/YA relative EKFs.
  It requires full position/velocity observations; recovery evidence is
  separate from maneuver identification and covariance calibration.
- Call out uncertainty, missing validation evidence, and model limits plainly.
- Treat output folders as derived evidence. Confirm that artifacts belong to
  the current config/run before citing them; rerun when provenance or freshness
  is uncertain.
- For performance changes, compare the same inputs before and after, require
  deterministic output parity at the appropriate tolerance, and run the
  applicable external validation workflow. Report timing separately from
  physics correctness and identify the measured hardware/configuration.
- When OEL itself appears hard to use, confusing, or missing an agent-facing
  workflow, follow `docs/agent-feedback-loop.md`: prepare a public-safe
  feedback draft, ask the user for permission, and submit only after approval.

The common agent loop is:

```text
request -> route -> scenario YAML or typed orbital-analysis problem -> validate
-> execute -> authoritative replay -> inspect -> bounded claim
```

Do not force a standalone analysis into scenario YAML, and do not invent an
MCP execution surface when the documented CLI or Python API is the supported
interface. For MCP-connected discovery, read `oel://analysis/workflows/v1`;
resource discovery is not execution authority.

## Code Navigation And Refactoring

- Treat established façade modules such as `sim.api`,
  `sim.config.scenario_yaml`, `sim.runtime_support`, `sim.master_outputs`,
  `sim.plotting.single_run`, and `sim.utils.plotting_capabilities` as stable
  import contracts. Use their ownership maps to locate implementation code.
- Put behavioral changes in the focused owner. Keep façades limited to
  compatibility exports, registries, dispatch, and explicitly documented
  downstream patch seams; do not grow a second implementation in a façade.
- Preserve existing names, signatures, class module paths, import identity,
  validation errors, ordering, serialization, CLI behavior, and artifact names
  unless the user explicitly approves a breaking change.
- When moving code, remove superseded duplicate or unreachable implementations
  after parity is established. One capability should have one authoritative
  implementation owner.
- Do not infer product availability from an implementation package or ownership
  registry. Product entitlements and export boundaries still govern which
  workflows may be recommended or executed.
- For architecture changes, run the applicable `test_*_architecture.py` tests,
  focused behavioral tests, `git diff --check`, and the public export integrity
  gate. Add newly public packages or docs to the allowlist manifest only after
  confirming they are public-safe.

## Scenario Generation Posture

Use a minimum viable scenario, then add complexity only when the user asks for
it or when it is necessary to answer the question they actually asked.

Ask a clarifying question when a missing detail changes the study:

- time horizon or duration,
- initial orbit, TLE, altitude, or relative state,
- passive vs controlled behavior,
- success metric or termination condition,
- fidelity level when the request says "realistic", "high fidelity",
  "operational", "deorbit", "decay", "access", or similar,
- whether the user wants plots, review-store inspection, or just summary
  outputs.

Default quietly when the detail is incidental:

- headless run,
- plots and animations off unless requested,
- attitude disabled unless attitude dynamics/control is part of the request,
- no sensing or estimation unless observation uncertainty, access, tracking, or
  closed-loop knowledge is needed,
- no Monte Carlo, sensitivity, campaign, optimizer, or report workflow unless
  requested,
- simple ONP dynamics first. Do not enable J2, J3, J4, drag, SRP, third
  bodies, spherical harmonics, or high-fidelity ONP propagation unless the user
  asks for them or the stated study requires them.

Examples:

- For "make a simple deorbit study", ask whether they want a simple maneuver
  geometry case or a drag-including decay study; do not silently enable a full
  perturbation stack.
- For "propagate this satellite for two hours", use simple propagation unless
  the user asks for perturbations or realistic force modeling.
- For "rendezvous with noisy measurements", sensing and estimation are
  relevant. For "rendezvous with perfect knowledge", do not add estimation.

## When The User Asks Something New

Do not force new user requests into the checked-in examples. Instead:

1. Restate the study goal in OEL terms.
2. Identify the minimum scenario elements needed: objects, initial state,
   dynamics, controller posture, duration, outputs, and review evidence.
3. Reuse public examples for structure when helpful, but give the new scenario
   a distinct name and output directory.
4. Validate before running, inspect artifacts after running, and explain what
   evidence supports the answer.
5. Name missing evidence or public-core limits instead of stretching an example
   beyond what it proves.

Use `docs/agent-capability-routing.md` to map broader requests such as TLE
propagation, access, whole-Earth coverage, directed-link budgets, attitude,
plotting, game/training, validation, sealed-mode, or comparison work to the
right public workflow and evidence.

## Review Query Workflow

The review store is the agent-friendly output inspection path. Use it when a
user asks questions about a completed run, wants tabular insight, or needs
custom figures from run evidence.

To create a queryable run, add this to scenario YAML before validation:

```yaml
outputs:
  review:
    enabled: true
    detail: standard
```

Then validate and run through the simulator:

```bash
python run_simulation.py --config <path> --validate-only
python run_simulation.py --config <path>
```

After the run, query the saved review DB:

```bash
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name, duration_s, samples FROM run_metadata"
python -m sim.review outputs/<scenario_name> --query "SELECT time_s, range_km FROM relative_state ORDER BY time_s LIMIT 20" --json
```

For a machine-readable evidence packet around common recipes or completed
outputs, use:

```bash
python -m sim.agent_task list
python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks
python -m sim.agent_task inspect outputs/<scenario_name> --query run_metadata --json
```

Rules for agents:

- Use only `SELECT` or `WITH` queries. The review API enforces read-only
  access; do not try to mutate, attach, or rewrite review databases.
- Query tables such as `run_metadata`, `objects`, `time_samples`,
  `object_state`, `relative_state`, `thrust`, `ground_access`,
  `coverage_summary`, `coverage_samples`, `coverage_intervals`,
  `coverage_transitions`, `link_summary`, `link_samples`, `link_windows`,
  `link_transitions`, `events`, `metrics`, and `artifacts` when present.
- Do not guess column names. Use `docs/agent-review-queries.md`,
  `review/schema.json`, or a small discovery query such as
  `SELECT * FROM object_state LIMIT 1` before writing custom SQL against an
  unfamiliar table.
- State the query used when summarizing a result so the user can reproduce the
  evidence.
- If `review/run.sqlite` is missing, fall back to `index.md`,
  `master_run_summary.json`, CSV histories, and plots. Do not pretend a review
  store exists.
- Use `sim.review.EvidencePlotter` or `python -m sim.review plot`
  for custom OEL-styled plots from completed review evidence. See
  `docs/agent-custom-plots.md`.
- When OEL MCP is connected, use `oel.plot_evidence.v1` for a matching
  supported recipe or the content-bound `oel.plan_review_plot.v1` then
  `oel.render_review_plot.v2` workflow for a custom review query. For any plot
  derived from OEL review evidence, use these OEL plotting surfaces before a
  host-native visualization tool. If OEL cannot express the requested figure,
  state the missing capability and use the documented custom-analysis fallback
  explicitly.
- For a supported movie derived from OEL review evidence, read
  `oel://review/animation-recipes/v1`, plan with
  `oel.plan_review_animation.v1`, and render the matching content-bound plan
  with `oel.render_review_animation.v1`. Inspect both the generated contact
  sheet and encoded movie before handoff. Do not substitute game recording or
  an ad hoc host animation for this review-evidence workflow.
- Before handing off a generated plot, visually inspect it and fix obvious
  presentation defects such as legends covering data, overlapping labels,
  clipped text, unreadable tick labels, blank figures, or missing expected
  series.
- Use `python -m sim.review` for table inspection and the custom
  plotting API for brief/report figures. Experimental review viewers and
  desktop workbenches are local-only until they are product-ready.

## Agent Feedback Loop

If an agent encounters a feedback-worthy OEL problem while helping a user, it
may propose submitting public feedback. Examples include missing examples,
confusing validation errors, insufficient artifacts for a reasonable public
question, unclear docs, or agent guidance conflicts.

Rules:

- Never submit feedback silently.
- Show the user the public-safe summary before submitting.
- Ask for explicit approval.
- Do not include secrets, API keys, customer data, CUI, export-controlled data,
  classified information, private configs, or private generated report packets.
- Use the private `SECURITY.md` process for vulnerabilities or sensitive-data
  exposure.
- Use the GitHub Agent Feedback issue template after approval.

## Public Commands

```bash
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
python run_simulation.py --config configs/automation_smoke.yaml
python -m sim.review outputs/<scenario_name> --query "SELECT scenario_name FROM run_metadata"
python run_game.py
```

For generated examples and evaluation fixtures:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

For repeatable public agent task checks, use `docs/agent-task-cards.md`.

## Safety And IP Boundary

- Only run scenario YAML from trusted sources. OEL configs can reference
  importable Python modules/classes.
- Keep API keys, proprietary configs, customer data, and generated report
  packets out of public commits.
- Public agents may explain public code. If a requested workflow depends on
  capability that is not included in the public core, say so and point to the
  documented public alternative.
