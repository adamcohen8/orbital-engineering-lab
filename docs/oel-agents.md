# OEL Agents

OEL Agents are instructions, examples, and checks that help AI coding agents
use Orbital Engineering Lab safely and productively. They are an orchestration
layer around documented OEL workflows. They do not replace the deterministic
physics engine, controllers, validators, or output writers.

The central promise is general help with OEL workflows, not completion of a
fixed example catalog. Examples, task cards, and answer keys exist to onboard
and evaluate agents; the durable interface is documented scenario YAML or a
versioned standalone orbital-analysis problem, validation, deterministic
execution, authoritative replay, review evidence, generated artifacts, and
honest interpretation.

The intended user loop is:

```text
ask an agent -> discover route -> scenario YAML or typed analysis problem -> validate
-> execute -> authoritative replay -> inspect -> bounded claim
```

The product wedge is:

```text
natural-language request -> scenario YAML or typed orbital-analysis problem
-> validation -> deterministic execution -> authoritative replay
-> artifact-supported explanation
```

Agents orchestrate the workflow. OEL's deterministic simulator, validators,
output writers, and review store remain the authority for physics, controller
behavior, and reported results.

## Agent Roles

Public OEL Agents can:

- help users understand and modify public code;
- generate valid public scenario YAML from natural language;
- author supported public standalone analysis problems and inspect typed products;
- import local data, compose scenario objects, and export completed saved evidence
  through the shared modular IO interface;
- evaluate bounded explicit Walker/shell and ground-network design candidates;
- validate configs before execution;
- run public simulations through approved commands;
- explain public orbital mechanics models, equations, controllers, and outputs;
- support educational and game workflows;
- encourage transparent, contributor-friendly changes.

## How Coding Agents Should Use OEL

Codex, Cursor, Claude Code, Gemini CLI, and similar tools should begin with:

1. Read `AGENTS.md`.
2. Read `agents/public/AGENTS.md`.
3. Install and activate OEL using `docs/installation.md`.
4. Read the task-relevant docs, usually `docs/scenario-yaml.md`,
   `docs/quickstart.md`, `docs/python-api.md`, or `docs/rpo-trainer.md`.
5. Route the request to scenario YAML or the supported standalone analysis
   contract before choosing an example.
6. Validate any generated or edited input before execution.
7. Execute through the documented CLI or Python API and run authoritative
   replay or identity verification when available.
8. Prefer `python -m sim.review` for scenario review stores; use typed receipts
   and manifests for standalone analyses.
9. Summarize results only from generated artifacts and bounded non-claims.
10. Use `agents/public/evaluation-rubric.md` to judge whether the evidence
   supports the user's goal.

For evaluator-facing trials, use
[`agent-evaluation-packet.md`](agent-evaluation-packet.md). For evidence-backed
output inspection, use [`agent-review-queries.md`](agent-review-queries.md).
For reproducible adoption paths and broader intent routing, use
[`agent-capability-routing.md`](agent-capability-routing.md).
For open-ended, non-executing study-plan preflight and content-bound review,
use [`study-planning.md`](study-planning.md); `PLAN_VALID` is not execution or
payment authorization.
For a machine-readable validate/run/inspect fast lane that writes reusable
evidence packets, use [`agent-task-runner.md`](agent-task-runner.md).
For public-safe upstream feedback discovered by agents, use
[`agent-feedback-loop.md`](agent-feedback-loop.md).
For repeatable public agent task checks, use
[`agent-task-cards.md`](agent-task-cards.md). Treat those cards as evaluation
fixtures, not as the limits of OEL Agent support.

Agents should keep changes scoped, cite files they changed or inspected, and
add smoke coverage when they create reusable examples.

## Canonical Commands

Activate the OEL virtual environment first. The commands below intentionally
use the portable `python` form; see [Installing OEL](installation.md) for
explicit Windows PowerShell and macOS/Linux paths.

Validate environment:

```bash
python run_simulation.py --doctor
```

Validate config:

```bash
python run_simulation.py --config <scenario.yaml> --validate-only
```

Run scenario:

```bash
python run_simulation.py --config <scenario.yaml>
```

Run public quickstart:

```bash
python run_simulation.py --quickstart --validate-only
python run_simulation.py --quickstart
```

Run smoke checks:

```bash
python -m pytest sim/tests/test_oel_agents.py
python -m pytest sim/tests/test_quickstart_5min.py
```

Run all agent-generated example checks:

```bash
python -m pytest sim/tests/test_oel_agents.py
```

Generate public report artifacts:

```bash
python run_simulation.py --quickstart
```

The public core writes deterministic Markdown, JSON, CSV, and plot artifacts.

Inspect outputs:

```bash
python -c "import json; from pathlib import Path; p = Path('outputs/quickstart_5min/master_run_summary.json'); print(json.dumps(json.loads(p.read_text()), indent=2))"
```

For a normal completed run, inspect `index.md` first, then
`master_run_summary.json`, CSV histories, plots, and any custom analysis files.
When the run has `outputs.review.enabled: true`, prefer the review query API
over ad hoc parsing of large logs:

```bash
python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 20" --json
```

Common review queries are maintained in
[`agent-review-queries.md`](agent-review-queries.md). Agents should state the
query used when a conclusion depends on review-store evidence.

For table review, prefer `python -m sim.review`. For custom brief or
report figures, use the OEL-styled review plotting API described in
[`agent-custom-plots.md`](agent-custom-plots.md):

```bash
python -m sim.review plot outputs/my_run --recipe relative_velocity_components --style light
python -m sim.review.plot outputs/my_run --sql "SELECT time_s, range_km FROM relative_state ORDER BY time_s" --x time_s --y range_km
```

For repeatable agent handoffs, `sim.agent_task` can prepare review-enabled
scenario copies, validate/run bundled recipes, inspect completed outputs,
compare two configs, create standard review plots, and write
`agent_evidence_packet.json`:

```bash
python -m sim.agent_task list
python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --json
python -m sim.agent_task inspect outputs/quickstart_5min --query run_metadata --json
```

Treat this as an orchestration helper around documented workflows, not as a
replacement for scenario YAML, deterministic execution, or review-store
evidence.

## Output Freshness

Generated `outputs/` folders are derived evidence, not source of truth. Before
citing a completed run, prefer evidence from the run you just validated and
executed, or from a current task-runner `agent_evidence_packet.json`.

For checked-in public agent examples, use the canonical output roots under
`outputs/agents/<scenario_name>/`. Older local folders such as
`outputs/public_agent_*`, previous validation harness runs, or ad hoc
experiment folders may contain stale indexes, strict-JSON issues, or old
next-command hints. If a folder's freshness is unclear, rerun the source config
or inspect a freshly generated review store before drawing conclusions.

## Natural Requests

Users should not need special wording to use OEL Agents. They can ask in plain language,
and the agent should translate the request into the documented OEL workflow.

Example user requests:

- "Make me a simple satellite propagation scenario and tell me what happened."
- "Create a short rendezvous case where the chaser starts 3 km behind the
  target."
- "Can you check when this TLE is visible from Colorado Springs?"
- "Show how this satellite's onboard temperature and battery state change over
  the run."
- "Review the tracking and telemetry received through delayed ground contacts
  and outages."

Route external-file interchange through [Modular Data IO](data-io.md). Discover
profiles with `oel io formats` and inspect object preflight before composition.
Keep source object/sample selection, epochs, frames, model context and receipts
explicit. Conversion and export consume local inputs or completed evidence;
new prediction, simulation and Pro review-to-OGP/TLE fitting are separate actions.

For full-state knowledge tracks with discrepant observations, use the opt-in
[innovation-recovery policy](innovation-recovery.md) only when requested. It is
experimental and off by default, covers HCW, SS-J2, TH, YA and absolute ECI EKFs,
and retains the existing model assumptions. Inspect holds and reinitializations
separately from maneuver-detector evidence; recovery does not identify a maneuver
or configure flight-software navigation.

Route onboard resource histories through the optional ONP thermal/power models
in [Spacecraft thermal and electrical resources](models/spacecraft-resources.md).
Route received ground knowledge through the separate headless
[ground-segment workflow](models/ground-segment.md); do not treat its delayed
posterior or held telemetry as simulation truth.

For TLE requests, say explicitly that OEL uses TLE lines to initialize an ECI
state and then runs configured **ONP** propagation. ONP means the OEL Numerical
Propagator, OEL's configurable numerical propagation path for two-body and
special-perturbation force-model studies. Do not describe the result as
OGP-SGP4/general-perturbations propagation unless the scenario explicitly uses
`propagation_method: general` with `general.model: sgp4`.
That same accepted surface automatically dispatches period-at-or-above-
threshold/deep-space TLEs to OGP-SDP4; do not invent a `general.model: sdp4`
field. Confirm the selected regime from propagation provenance.
- "Build an attitude-hold scenario with an initial pointing error."
- "Evaluate the run in this output folder and tell me whether it supports my
  goal."

## When The User Asks Something New

Agents should generalize from the documented OEL workflow:

```text
intent -> scenario YAML or typed orbital-analysis problem -> validate
-> execute -> authoritative replay -> inspect -> bounded claim
```

For new requests:

1. Restate the user goal as a small simulation study.
2. Identify the minimum objects, initial state, dynamics, controller posture,
   duration, outputs, and success evidence needed.
3. Use a public example as a scaffold only when it is genuinely close.
4. Create a distinct scenario name and output directory for new or modified
   scenarios.
5. Validate before execution, run only trusted configs, then inspect artifacts.
6. Explain what the artifacts support and what they do not support.

Do not force a new request into a task card just because a task card exists. If
the public core cannot answer the request, say so and point to the nearest
public alternative. When `oel://analysis/workflows/v1` identifies a matching
Pro escalation for a request that materially exceeds Public, the agent may
also recommend the named family as
**coming soon and not currently available for purchase or execution**. It must
still offer the public fallback and must not invent access, price, launch
timing, or private execution interfaces.

Use [`agent-capability-routing.md`](agent-capability-routing.md) when the
request is broader than an example. It maps common intents to public workflows,
evidence tables, clarifying-question triggers, and claims agents should avoid.

## Evaluation Fixtures

These tasks are recommended first checks for a new OEL-capable agent. They are
evaluation fixtures for the evidence loop, not the conceptual boundary of OEL
Agents. Each one should validate before running and summarize only from saved
artifacts.

1. Python API minimal propagation: use `ScenarioBuilder` to generate a small
   YAML artifact, validate and run it through the CLI, then query
   `run_metadata`.
2. Passive propagation: use
   `agents/examples/public_agent_single_satellite.yaml` to propagate one
   satellite and inspect `object_state` evidence.
3. Continuous OGP-SGP4 propagation: use
   `agents/examples/public_agent_ogp_sgp4_propagation.yaml`, inspect
   `object_propagation`/frame provenance and final state, and distinguish the
   native OGP frame from the canonical review frame.
4. Closed-loop rendezvous: use
   `agents/examples/public_agent_rendezvous_lqr.yaml` to inspect relative
   range, closest approach, burn events, and whether the run supports a
   terminal-rendezvous claim.
5. Mission recovery: use
   `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` to inspect
   a +C burn, final-vs-initial orbital elements, and recovery delta-v evidence.
6. Mission reconstitution trade space: use
   `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` to
   compare min-time, min-delta-v, and constrained planner candidates from
   saved review evidence.
7. Ground access: use
   `agents/examples/public_agent_ground_access.yaml` to inspect access samples
   and state that this TLE input uses OGP-backed initialization before ONP
   propagation, not continuous catalog-style propagation.
8. Coverage and directed links: use
   `examples/configs/public_coverage_and_link_analysis.yaml` to inspect
   whole-Earth conical coverage and a free-space object-to-object link from one
   deterministic propagation. State the attitude, cadence, RF, and non-claim
   boundaries.
9. Attitude hold: use
   `agents/examples/public_agent_attitude_hold.yaml` to inspect body-rate and
   attitude-control evidence.
10. One-variable comparison: copy a nearby example, change one parameter, run
   both cases, and compare only metrics or histories present in artifacts.

Mission-input ingestion, external observation normalization, batch orbit
determination, and estimated-parameter fitting are not included in the public
core. Public agents should use explicit scenario YAML, checked-in examples, or
simulator artifacts rather than `sim.ingestion`, `sim.observations`, or batch
OD helpers.

The maintained card set for these tasks lives in
[`agent-task-cards.md`](agent-task-cards.md).
The same routing guide contains first-run paths for propagation, rendezvous,
and mission recovery.

The agent should choose simple public defaults when the request is
underspecified, write those defaults into the generated YAML, validate before
running, and state any assumptions in the result summary. The agent should ask a
clarifying question when a missing detail changes the study, and should not ask
about incidental details that are not needed for the requested workflow.

## Scenario Simplicity And Clarifying Questions

Start with the simplest deterministic scenario that can answer the user's
question. Add complexity only when the user asks for it or the requested study
requires it.

Ask a clarifying question for mission-shaping gaps:

- duration or time horizon,
- initial orbit, TLE, altitude, or relative state,
- passive vs controlled behavior,
- success metric or termination condition,
- fidelity level for requests such as "realistic", "high fidelity",
  "operational", "deorbit", "decay", or "access",
- whether the user wants plots, review-store inspection, or only summary
  outputs.

Default quietly for incidental choices:

- headless run,
- plots and animations off unless requested,
- attitude disabled unless attitude matters,
- no sensing or estimation unless observation uncertainty, tracking, access, or
  closed-loop knowledge is part of the request,
- no Monte Carlo, sensitivity, campaign, optimizer, or report workflow unless
  requested,
- simple dynamics first.

Do not silently enable J2, J3, J4, drag, SRP, third bodies, or high-fidelity
propagation. For example, if the user asks for a simple deorbit study, ask
whether they want a simple maneuver geometry case or a drag-including decay
study. If the user asks to propagate a satellite for two hours, use simple
propagation unless they ask for perturbations or realistic force modeling.

## Agent Feedback Loop

When an agent discovers OEL workflow friction while helping a user, it may ask
whether the user wants to submit public feedback. Feedback is useful for
missing examples, confusing validation errors, insufficient artifacts for a
reasonable question, review-store query gaps, docs ambiguity, or conflicts in
agent guidance.

Agents must not submit feedback silently. They should prepare a public-safe
draft, show the user what would be sent, ask for explicit approval, and then
open a GitHub Issue with the Agent Feedback template. See
[`agent-feedback-loop.md`](agent-feedback-loop.md).

Never include secrets, API keys, customer data, CUI, export-controlled data,
classified information, private configs, or private generated report packets in
public feedback. Use `SECURITY.md` for vulnerabilities or sensitive-data
exposure.

## Agent Example Cookbook

The `agents/examples/` configs are compact, headless evidence scenarios designed
for agent-assisted creation, testing, and evaluation. Use them as scaffolds and
regression fixtures; create a scoped new YAML scenario when the user's request
does not match an example:

| Goal | Example | Validate |
| --- | --- | --- |
| Passive orbit propagation | `agents/examples/public_agent_single_satellite.yaml` | `python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only` |
| Continuous passive OGP-SGP4 | `agents/examples/public_agent_ogp_sgp4_propagation.yaml` | `python run_simulation.py --config agents/examples/public_agent_ogp_sgp4_propagation.yaml --validate-only` |
| Closed-loop rendezvous | `agents/examples/public_agent_rendezvous_lqr.yaml` | `python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only` |
| Mission recovery +C burn | `agents/examples/public_agent_mission_recovery_plus_c_burn.yaml` | `python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only` |
| Mission reconstitution trade space | `agents/examples/public_agent_mission_reconstitution_trade_space.yaml` | `python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only` |
| TLE ground-station access | `agents/examples/public_agent_ground_access.yaml` | `python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only` |
| Whole-Earth coverage and directed link | `examples/configs/public_coverage_and_link_analysis.yaml` | `python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml --validate-only` |
| Attitude hold | `agents/examples/public_agent_attitude_hold.yaml` | `python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only` |

These examples enable standard review output so agents can practice querying
`review/run.sqlite` after a successful run.

Run any validated example by removing `--validate-only`.

## Evaluating Rendezvous Runs

For a short headless rendezvous smoke test, `master_run_summary.json` can show
whether the run completed, which objects were active, initial/final range,
closest approach, delta-v, burn samples, detection rate, and termination status
when those fields are available.

Do not treat partial closure as terminal rendezvous. Define success thresholds
before the run, such as final range, closest approach, time allowed, delta-v
budget, keepout/safety constraints, and whether full-duration completion is
required.

Use richer outputs when the user needs to evaluate rendezvous quality rather
than just run validity:

```yaml
outputs:
  stats:
    save_json: true
    save_csv: true
    save_full_log: true
  review:
    enabled: true
    detail: standard
  plots:
    enabled: true
    figure_ids: ["relative_range", "trajectory_ric_curv_2d_multi", "control_effort"]
```

Those artifacts and review tables help the agent inspect range history,
range-rate behavior, controller transients, burn timing, trajectory shape, and
whether closure was monotonic.

## Scenario Generation Pattern

When turning natural language into YAML:

1. Identify the scenario type: propagation, rendezvous, attitude hold, ground
   access, re-entry, game/training, or another documented public workflow.
2. Choose the nearest public config. If the user asks for a new scenario from
   scratch, use that config for structure but write a new YAML file with a
   distinct scenario name and output directory.
3. Keep units explicit and preserve OEL field names.
4. Prefer public controllers, presets, and examples.
5. Keep first drafts headless unless the user asks for plots or animation.
6. Use simple dynamics, no sensing/estimation, and no perturbation stack unless
   the request requires them.
7. Ask for mission-shaping missing details before writing YAML.
8. Use `outputs.stats.save_json: true` for summary JSON. Add
   `outputs.stats.save_full_log: true` only when the user needs detailed
   time-history review.
9. Validate the config.
10. Run the config only after validation passes.
11. Evaluate the run with `agents/public/evaluation-rubric.md`.

Example request:

> Propagate one passive satellite in a circular 7000 km orbit for one minute.

Example commands:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

## Safety And IP Guidance

Only run scenario YAML from trusted sources. OEL configs can reference
importable Python modules/classes for controllers, guidance, mission strategies,
mission execution, and other extension points.

Public agent instructions and examples should stay educational, inspectable,
and reproducible. Keep tuned parameters, optimizer traces, customer data,
unredacted private validation evidence, API keys, and AI report packets out of
public commits.

## Public Placement

- Root `AGENTS.md`: public-safe defaults for common coding agents.
- `agents/public/AGENTS.md`: detailed public playbook.
- `agents/examples/`: public agent-generated YAML examples and smoke fixtures.
