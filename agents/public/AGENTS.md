# Public OEL Agents

Start with the shared [operational bootstrap](../../sim/installation/data/agent-bootstrap.md).
MCP serves the identical guide at `oel://agent/bootstrap/v1` and compact active
routes at `oel://agent/workflows/v1`. The [study planner](../../docs/study-planning.md)
is a separate prototype; its tools are usable only when listed by the connection.

Use this playbook when an AI coding agent is helping with the open-source
Orbital Engineering Lab core.

## Mission

Public OEL Agents make the project easier to understand, modify, and run. They
can help users generate valid public scenario YAML or typed standalone
orbital-analysis problems, validate inputs, run approved workflows, replay and
inspect evidence, explain public models/controllers, and support educational
or game workflows.

Agents should be transparent contributors. They should cite the files or docs
they used, prefer public examples over invention, and leave deterministic
simulation and reporting to OEL itself.

Examples are rails, not boundaries. Use them when they fit the user's request,
but do not reduce OEL Agents to a fixed example catalog. The durable agent
interface is the documented workflow: scenario YAML or a versioned analysis
problem, validation, deterministic execution, authoritative replay, review
queries, artifacts, tests, and honest limits.

Terminology: **OGP** means the **OEL General Propagator**, OEL's catalog-style
general-perturbations family for TLE/mean-element products. **OGP-SGP4** is the
supported near-Earth SGP4 path; **OGP-SDP4** is the supported deep-space
SDP4/resonance path. **ONP** means the **OEL Numerical Propagator**, OEL's configurable
numerical propagation path for two-body and special-perturbation force-model
studies. Do not call ONP HPOP; reserve HPOP for external
reference/validation workflows.

## Public Execution Boundary

- Public single-scenario workflows use deterministic serial object stepping.
  Do not enable or recommend automatic/process-pool object parallelism in a
  public config.
- Thread transports for object steps, campaigns, controller studies, and
  Scale remain Pro workflows; an available transport does not change the
  public execution boundary or supply a missing entitlement.
- Automatic within-scenario object workers, hierarchical campaign/object
  planning, Monte Carlo, sensitivity, config queues, and controller benchmarks
  are outside the public core. Do not reproduce them with ad hoc public scripts.
- The public `sim.analysis.coverage_sensitivity` helper compares explicit,
  already-produced deterministic coverage products. It is not the automatic
  sensitivity-campaign executor described above.
- When such a workflow is unavailable, use the closest deterministic public
  alternative: one validated run, explicit paired runs, or a small manually
  enumerated set whose assumptions and evidence remain inspectable.
- When the user explicitly intends Hosted OEL Pro execution, public OEL may
  prepare and validate the public package contract in
  `docs/hosted-pro-package.md`. This plans a published Pro capability and binds
  selected bytes; it does not reproduce the Pro executor locally.
- Treat `CLIENT_VALID` as permission to request authoritative Hosted preflight
  only. It is not proof of availability, entitlement, price, payment approval,
  scientific qualification, or execution authority.

## Supported Workflows

1. Install and activate OEL using `docs/installation.md`.
2. Read `docs/agent-capability-routing.md` and the relevant workflow docs.
3. Choose scenario YAML for simulator time histories, or the documented
   versioned problem/product contract for a standalone orbital analysis.
4. Validate the selected input before execution.
   For scenario YAML, validate before running it.
5. Execute only through the documented CLI or Python API.
6. Run the workflow's authoritative replay or identity verification when it
   exists; replay is separate from solver success.
7. Inspect generated evidence, including review queries for scenario runs and
   typed receipts/manifests for standalone analyses.
8. Evaluate the result with `agents/public/evaluation-rubric.md`.
9. Summarize results from saved artifacts, not from memory or speculation.
10. Add tests or smoke checks for new agent-facing examples.

For an open-ended question that needs a reviewable operation graph before any
execution, use the non-executing `oel study` workflow in
`docs/study-planning.md`. `PLAN_VALID` means only that deterministic preflight
accepted the proposal. It does not approve execution or payment, entitle a Pro
capability, or let the agent approve its own plan. Stop at plan review unless a
separate documented execution surface and explicit user authorization exist.

When the user wants Hosted OEL to execute an analysis, an agent may prepare
the public package format in `docs/hosted-pro-package.md` and run
`oel hosted validate-package <package> --workspace-root <root>`. Treat
`CLIENT_VALID` as local package-integrity and planning evidence only. It does
not establish remote availability or entitlement, upload files, create a quote,
authorize payment, or execute anything. A public-only package may run locally
for free or receive an optional Hosted quote; a Pro package cannot run in the
public installation. Hosted OEL must revalidate the exact sealed bytes before
offering execution terms.

For local external formats, shared object composition, and exports from
completed evidence, use `docs/data-io.md` and `oel io` (`python -m sim.data_io`
in a source checkout). Discover actual format directions and optional backends
with `oel io formats`; use `object describe-import` or `object describe-export`
to inspect requirements, retained fields, defaults, and promotion gates.
These are bounded offline engineering profiles, not comprehensive standards
implementation or operational qualification. Use explicitly supplied EOP or
SPICE kernels when the selected frame/time conversion requires them.

IO operations do not execute simulations, rerun OD, or replace source inputs.
Select exact objects and samples; retain frames, absolute epochs, units and
adjacent hash-bound receipts. An imported trajectory sample creates an orbital
seed, while native checkpoints/snapshots retain their separate continuation
contracts. Compose into a new scenario/output location and validate it before
execution. Export a predicted trajectory only after a separate prediction run.
Public conversion and export do not fit review trajectories to OGP/TLE products;
that fitting remains a separate Pro workflow.

For a study that needs full-state EKF anomaly recovery, consult
`docs/innovation-recovery.md`. The shared absolute and HCW/SS-J2/TH/YA relative
policy is experimental and off by default; enable it only for the study's
explicit full position/velocity measurement contract. Recovery holds and
reinitializations do not establish a maneuver, calibrated covariance, sensor
accuracy, or onboard navigation qualification.

For whole-Earth coverage or directed-link questions, start with
`docs/coverage-link-scenario-analysis.md` and
`examples/configs/public_coverage_and_link_analysis.yaml`. Treat the analysis
as evidence-only post-processing unless the user explicitly requests the
bounded authorized runtime monitor. Use the coverage/link review tables and
state the geometry, cadence, attitude, terminal, and environmental assumptions;
do not promote geometric visibility or a free-space margin into operational
sensor or communications assurance.

For headless ground tracking and delayed telemetry, use the optional top-level
`ground_segment` scenario layer documented in `docs/models/ground-segment.md`.
Keep its posterior and received/held channels separate from simulation truth;
orbit estimation needs an explicit prior, and prediction is two-body. This is a
reusable simulation capability, not an Operator Trainer interface.

For time histories of onboard temperature or electrical resources, see
`docs/models/spacecraft-resources.md`. Enable `specs.thermal` and `specs.power`
only when the study needs them; their outputs are simulation truth and do not
become telemetry or navigation belief unless a separate observation boundary
is configured.

For custom complete-stack flight software, use the bounded Public FSW
Authoring Kit in `docs/fsw-authoring.md`. Inspect unfamiliar candidate material
without importing it, obtain explicit source trust before lifecycle validation,
and use only the declared component suite plus one deterministic serial smoke.
Do not recreate private Controller Bench, tuning, qualification, packaging,
external-process, or cFS/SIL workflows in public scripts.

The preferred agent evidence loop is:

```text
request -> route -> scenario YAML or typed orbital-analysis problem -> validate
-> execute -> authoritative replay -> inspect -> bounded claim
```

Public standalone analysis includes bounded CCSDS interchange, canonical
frame/time conversion, trajectory targeting, conjunction assessment,
collection analysis, tracking-data OD, mission scheduling, bounded
constellation design, spacecraft power, orbit lifetime, and the content-bound
study lifecycle. Use
`oel://analysis/workflows/v1` for MCP-connected discovery. Only the study,
CCSDS, and frame/time read-only adapters are currently exposed through MCP;
use the documented CLI or Python API for the other public workflows.

That resource may also name a recommendation-only Pro escalation when the
user's request materially exceeds a public workflow. Describe it as
**coming soon and not currently available for purchase or execution**, and
offer the listed public fallback. Do not treat the metadata as entitlement or
execution authority, invent a launch date or price, expose private interfaces,
or upsell Pro when the public workflow is sufficient.

For a small explicit Walker/shell and ground-site trade, use
`docs/constellation-design.md` and `python -m sim.constellation_design`. Keep
the public workflow inside its declared candidate and work bounds. Do not turn
it into automatic design-space generation, large optimization, ground-site
placement, crosslink routing, capacity planning, or a managed campaign; those
remain excluded private or future workflows.

Use `docs/agent-evaluation-packet.md` to evaluate whether an agent follows this
loop. Use `docs/agent-capability-routing.md` for first-run paths and broader
intent routing, including evidence, questions, and public-core limits. Use
`docs/study-planning.md` for the restricted non-executing BYO-agent planning
loop. Use
`docs/agent-task-runner.md` when a repeatable recipe, comparison, plot, or
portable `agent_evidence_packet.json` would help another agent inspect the same
evidence. Use `docs/agent-review-queries.md` for reusable review-store SQL. Use
`docs/agent-feedback-loop.md` when an agent finds public-safe workflow feedback
worth sending upstream. Use `docs/agent-task-cards.md` for the repeatable
public agent task-card set; task cards are evaluation fixtures, not the
boundary of what agents can help users do.

For a long trusted single-scenario run, use `docs/agent-run-lifecycle.md`.
Preserve the accepted handle, perform a bounded await, and inspect the durable
terminal manifest before reading simulation evidence. Treat wake notification
as a reason to inspect committed state, never as evidence that the physics run
passed. If await reports `owner_lost`, use the explicit `reconcile` operation
to commit `interrupted`; await itself remains read-only. Do not use the
lifecycle as an arbitrary command runner.

## Public Code Navigation

OEL keeps established import modules as compatibility façades while focused
packages own implementation families. Start with the supported façade, then
use these public-safe maps before editing code:

- `docs/config-api-architecture.md` for scenario configuration and `sim.api`;
- `docs/runtime-architecture.md` for runtime construction and single-run
  collaborators;
- the implementation map in `docs/plotting.md` for output and plotting
  families.

External callers and examples should continue to import the stable façade.
Maintainers should change the focused owner, preserve the façade export and
class/import identity, and avoid copying implementation logic back into the
façade. Remove obsolete extraction copies once parity is established so an
agent cannot edit an apparently authoritative module while production still
uses a duplicate elsewhere.

An ownership map is a navigation aid, not proof that an excluded capability is
available publicly. Keep the Public Execution Boundary above in force. When a
public implementation package or architecture document is added, update the
public surface manifest deliberately and run the exported-tree integrity and
public test gates.

## Canonical Commands

Activate the OEL environment first. Commands below use `python` on every
supported OS; `docs/installation.md` provides explicit PowerShell and POSIX
interpreter paths when activation is unavailable.

Validate environment:

```bash
python run_simulation.py --doctor
```

Validate config:

```bash
python run_simulation.py --config configs/automation_smoke.yaml --validate-only
```

Run scenario:

```bash
python run_simulation.py --config configs/automation_smoke.yaml
```

Run smoke test:

```bash
python -m pytest sim/tests/test_oel_agents.py
```

Start and await a trusted foreground run:

```bash
oel runs start --config configs/automation_smoke.yaml --output-dir agent-smoke --jsonl
oel runs await RUN_ID --timeout 900
```

Scaffold and inspect a public FSW candidate:

```bash
oel fsw init my_adcs --template adcs
oel fsw inspect fsw_candidates/my_adcs/candidate.yaml
```

Generate a report-like public artifact set:

```bash
python run_simulation.py --quickstart
```

Inspect outputs:

```bash
python -c "import json; from pathlib import Path; p = Path('outputs/quickstart_5min/master_run_summary.json'); print(json.dumps(json.loads(p.read_text()), indent=2)[:4000])"
```

Public AI report generation is not a default open-source workflow. In the
public core, treat `index.md`, JSON summaries, CSV files, plots, and game
debriefs as report artifacts.

Inspect review stores when enabled:

```bash
python -m sim.review outputs/my_run --query "SELECT scenario_name, duration_s FROM run_metadata"
python -m sim.review outputs/my_run --query "SELECT time_s, range_km FROM relative_state LIMIT 20" --json
```

Package a repeatable recipe or completed-run inspection:

```bash
python -m sim.agent_task list
python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks
python -m sim.agent_task inspect outputs/my_run --query run_metadata --json
```

Inspect local IO requirements, create a new passive study, and export a
selected completed state:

```bash
oel io formats
oel io object describe-import initial.opm --format opm-kvn
oel io object import initial.opm configs/imported.yaml --format opm-kvn --as target --scenario-name imported --run-output outputs/imported --duration-s 600 --step-s 10
python run_simulation.py --config configs/imported.yaml --validate-only
oel io object describe-export outputs/my_run --object-id target
oel io object export outputs/my_run final.opm --format opm-kvn --object-id target --final
```

The export commands require a separately completed, provenance-verified run;
they do not run the imported scenario. Use `oel io export-review` for saved
trajectories/attitude histories and `oel io export-table` for a bounded read-only
query against the source run's actual schema.

To continue from one exact completed-run state, export a versioned product and
then materialize a separate validated ONP scenario:

```bash
python -m sim.handoff export-state outputs/my_run --object-id target --sample final --output outputs/my_run_final_state.json
python -m sim.handoff materialize-onp --state-product outputs/my_run_final_state.json --scenario-name my_continuation --output outputs/my_continuation.yaml --run-output-dir outputs/my_continuation --duration-s 600 --dt-s 10
python -m sim.handoff compare-handoff --product outputs/my_run_final_state.json --scenario outputs/my_continuation.yaml --output outputs/my_continuation.comparison.json
```

The source run must have review output, a canonical ECI state frame, and an
absolute initial epoch. These commands select and validate evidence but do not
execute the continuation.
After a separately authorized run, pass `--run-output-dir` to compare the first
consumer review row with the promoted state.

When using the review store, cite the SQL query or saved view that supports the
answer. Keep queries read-only, prefer `SELECT`/`WITH`, and inspect table names
or sample rows before writing custom SQL against an unfamiliar table:

```bash
python -m sim.review outputs/my_run --query "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name" --json
python -m sim.review outputs/my_run --query "SELECT * FROM object_state LIMIT 1" --json
```

You can also inspect table names with the Python API:

```python
from sim.review import ReviewWorkspace

workspace = ReviewWorkspace.open("outputs/my_run")
print(workspace.tables())
print(workspace.schema()["columns"].keys())
```

Use review tables for time-history questions, range closure, burn activity,
ground-access checks, metrics, and artifact inventories. If `review/run.sqlite`
is missing, fall back to `index.md`, `master_run_summary.json`, CSV histories,
and plots without claiming structured review evidence exists.

## Scenario Generation Rules

- Start with the simplest deterministic scenario that can answer the user's
  question. Do not add unrequested physics, sensors, estimators, controllers,
  plots, animations, or campaign machinery.
- Start from a nearby public example whenever possible. When the user asks for
  a new scenario from scratch, use examples for structure but write a new YAML
  file with a distinct scenario name and output directory.
- Use `objects:` entries for spacecraft instead of legacy top-level `chaser`
  or `target` sections.
- Keep units explicit in field names: `_km`, `_s`, `_deg`, `_rad_s`,
  `_km_s`, `_kg`, `_n`, and similar suffixes.
- Keep first drafts short, headless, and deterministic: plots off, animations
  off, modest duration, and public controllers.
- Use simple ONP dynamics first. Do not enable J2, J3, J4, drag, SRP, third
  bodies, high-fidelity ONP propagation, sensing, estimation, Monte Carlo,
  sensitivity, or reports unless the user asks for them or they are necessary
  for the stated study.
- For JSON outputs, use `outputs.stats.save_json: true` for the compact
  `master_run_summary.json`. Set `outputs.stats.save_full_log: true` only when
  the user needs detailed time-history review in `master_run_log.json`.
- Validate every generated config with `--validate-only`.
- Run only trusted YAML. Scenario plugin pointers can import Python code. Run
  `--safe-validate` first for user-provided or unfamiliar YAML. Safe validation
  inspects the config without importing plugin surfaces; it does not make the
  config safe to execute. Use ordinary `--validate-only` and execution only
  after referenced plugins, modules, and paths are trusted.

Ask a clarifying question when a missing detail changes the study: duration,
initial orbit/TLE/relative state, passive vs controlled behavior, success
metric, termination condition, or fidelity level. Do not ask about incidental
details that are not needed for the requested workflow. For example, do not ask
about sensing/estimation unless observation uncertainty, tracking, access, or
closed-loop knowledge is part of the request.

For deorbit, decay, access, or "realistic" requests, clarify the intended
fidelity instead of silently enabling every force model. A simple deorbit
maneuver geometry demo and a drag-including orbital decay study are different
configs.

## When The User Asks Something New

Agents should generalize from OEL's documented interfaces, not from memorized
example outcomes. For a request that is not exactly one of the checked-in
examples:

1. Translate the request into the minimum OEL workflow and acceptance evidence.
2. Decide whether it needs simulator time history and scenario YAML, or a
   supported versioned standalone analysis problem/product.
3. Reuse an example only when its contract genuinely fits the request.
4. Validate the selected input before execution.
5. Execute through the documented interface and retain the complete evidence.
6. Perform authoritative replay or identity verification when available, then
   inspect review tables, typed receipts, manifests, and artifacts.
7. State assumptions, missing evidence, non-claims, and model limits.

If no public-core workflow can answer the request, say so and point to the
nearest public alternative instead of pretending an example proves more than it
does.

Use `docs/agent-capability-routing.md` before choosing an example, customizing
a first-run scenario, or authoring a standalone problem. It covers workflow evidence and clarifying-question
triggers for propagation, TLEs, rendezvous, access, attitude, plotting, game
training, comparison, validation, sealed mode, and public-core boundaries.

## Rendezvous Evaluation Notes

For a closed-loop rendezvous scenario, compact summary JSON is enough for a
first smoke check when it includes initial range, closest approach, final
range, delta-v, burn samples, detection rate, and termination status. Treat
"validated and ran" as different from "rendezvous succeeded."

Before claiming success, define the user's range, time, delta-v, and safety
thresholds. A run that closes distance but ends kilometers away demonstrates
controlled approach, not terminal rendezvous.

Enable richer artifacts when the user needs trajectory-quality review:

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

Use those richer artifacts to inspect range history, range-rate behavior,
controller transients, burn timing, trajectory shape, and whether closure was
monotonic or merely improved at the final sample.

## Example Rails

Request:

> Propagate one passive satellite in a 7000 km circular, 45 degree orbit for
> one minute using two-body dynamics, no plots, and saved JSON stats.

Use:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml
```

Request:

> Give me a quick closed-loop rendezvous example with public controllers.

Use:

```bash
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml --validate-only
python run_simulation.py --config examples/configs/public_closed_loop_rendezvous_lqr.yaml
```

Request:

> Help me practice manual RPO.

Use:

```bash
python run_game.py examples/configs/public_manual_rpo_training.yaml
```

## Natural User Requests

Users should be able to ask for OEL work in ordinary language. Do not require
them to provide a structured template. Interpret the request, choose the
nearest public workflow, create or edit scenario YAML or a typed analysis
problem, validate, execute, replay, inspect evidence, and evaluate the result.

Examples of acceptable user requests:

- "Make me a simple satellite propagation scenario and tell me what happened."
- "Create a short rendezvous case where the chaser starts 3 km behind the
  target."
- "Can you check when this TLE is visible from Colorado Springs?"

For TLE requests, inspect and state the configured propagation contract. A TLE
may initialize an ECI-compatible state followed by configured ONP propagation,
or it may drive continuous passive OGP propagation when the scenario selects
`propagation_method: general` with `general.model: sgp4`; deep-space mean
elements dispatch to OGP-SDP4 automatically. Do not infer
one path from the presence of TLE lines; use the normalized config and saved
propagation provenance.
- "Build an attitude-hold scenario with an initial pointing error."
- "Evaluate the run in this output folder and tell me whether it supports my
  goal."

If the request leaves out necessary details, make conservative public defaults
visible in the generated YAML and explain them in the result summary. Ask a
clarifying question only when a reasonable public default would be misleading
or risky.

## Agent-Generated Example Configs

These examples are intentionally short and headless so users can validate,
run, and inspect them quickly. They are useful starting points and regression
fixtures, not a complete list of supported agent workflows:

```bash
python run_simulation.py --config agents/examples/public_agent_single_satellite.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_rendezvous_lqr.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_mission_recovery_plus_c_burn.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_mission_reconstitution_trade_space.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_ground_access.yaml --validate-only
python run_simulation.py --config agents/examples/public_agent_attitude_hold.yaml --validate-only
```

These examples enable standard review output. After running one, inspect
the configured output directory's `review/run.sqlite` with
`python -m sim.review` and cite the query used for any important
metric.

For a repeatable evaluation set around these examples, use the task cards in
`docs/agent-task-cards.md`.

## Explanation Rules

- Explain public equations, frames, controls, outputs, and limitations from
  checked-in docs and source.
- Distinguish model behavior from real mission behavior.
- Do not claim validation beyond the artifacts and tests that actually exist.
- When a result matters, point the user to the generated output file and the
  command that produced it.

## Agent Feedback Loop

Agents may help users submit public-safe feedback about OEL itself when a
workflow problem appears during normal use. Feedback-worthy issues include
missing examples, confusing validation messages, docs gaps, review-store
limitations, insufficient output artifacts, or conflicts in agent guidance.

Follow `docs/agent-feedback-loop.md`:

- prepare a short public-safe feedback draft,
- show the user what would be sent,
- ask for explicit approval,
- submit only after approval using the GitHub Agent Feedback issue template,
- link the issue back to the user.

Do not use public feedback for vulnerabilities, secrets, customer data, CUI,
export-controlled data, classified information, private configs, or private
generated report packets. Use the private `SECURITY.md` process instead.

## Safety And IP Guidance

- Do not commit API keys, local customer data, generated AI report packets, or
  non-public outputs.
- Do not copy tuned parameters, optimizer traces, customer material, or
  generated validation evidence into public examples.
- Keep public examples educational, inspectable, and reproducible.
- If a user asks for capability that is not present in the public core,
  redirect to documented public CLI/API behavior and examples.
