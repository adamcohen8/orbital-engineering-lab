# Operating OEL through an agent

Start with the user's question. You own OEL's commands, configuration and
evidence workflow; the user supplies material study choices. Ask only when a
missing choice changes the answer. Examples are scaffolds, not a fixed catalog.
Use the deterministic OEL engine; never substitute agent-written physics.

## Discover the active route

- With MCP, call `oel.describe_capabilities.v1`, then read
  `oel://agent/workflows/v1`. Readiness describes process configuration only;
  registered tools, configured approvals, input trust and execution authority
  are different things. Never invent approval IDs or change data markings to
  bypass a denial. Reuse applicable operator-provided approvals within scope.
- If `oel.study.capabilities.v1` is listed, use that catalog and
  `oel.study.preflight.v1` for a typed proposal. Preserve
  `CLARIFICATION_REQUIRED` and `UNSUPPORTED`; `PLAN_VALID` never authorizes a
  run. The study planner is a separate local prototype (`oel-study-mcp`), not
  an execution tool implicitly present on every OEL connection. A host operator
  must configure that connection when needed. Do not require it for ordinary
  documented single-scenario work.
- With local command access, use `oel doctor` and `oel workspace status .`.
  Managed commands use `oel --workspace . sim ...` and
  `oel --workspace . review ...`. A source checkout also supports its installed
  Python with `run_simulation.py` and `python -m sim.review`.
- A capability's implementation or declared maturity is not proof of current
  availability, entitlement or qualification. Use the active catalog and
  evidence. If a route is unavailable, explain the limit before offering an
  alternative that changes the study.

Public scenarios use deterministic serial object stepping. Scale, campaigns,
controller studies, and their opt-in thread/process transports remain Pro; do
not recreate those workflows in agent scripts or infer entitlement from code.

## Construct and check the study

1. Identify the objects, initial states, duration, physical assumptions,
   control/navigation posture and success evidence needed for the question.
   Keep frames, epochs and units explicit. Ask about consequential ambiguities;
   choose incidental settings without burdening the user.
2. Author a distinct scenario through documented YAML or `sim.api.ScenarioBuilder`
   using the host's authorized file tools. Main MCP planning takes an existing
   `config_path`; it is not a general file editor. Product materialization and
   FSW scaffolding cover their documented special cases only.
3. Default to headless execution, review enabled, plots/animations off, and the
   simplest dynamics that answer the question. Add perturbations, sensing,
   estimation or campaigns only when required. For MCP-compatible evidence use
   `outputs.stats.save_full_log: false` and compact/standard review detail.
   If full detail is necessary, identify an authorized compatible route instead
   of silently dropping required evidence.
4. Inspect unfamiliar YAML with `--safe-validate` before importing validation.
   Unknown fields are intent failures: consult the schema instead of deleting
   settings to make validation pass. Check that the resolved configuration still
   answers the question; structural validity is not scientific adequacy.

## Import, compose, or export local data

Use `oel io formats` and the bounded offline profiles in `docs/data-io.md`.
Preflight source requirements with `oel io object describe-import` and the
required source/format arguments; inspect completed-output routes with
`oel io object describe-export` and an explicit object selector. The
source-checkout entry point is `python -m sim.data_io`. These are CLI/Python
routes; do not invent corresponding MCP tools.

Keep object/sample selection, frames, epochs, units, promotion gates and
hash-bound receipts explicit. Imports and composition write a new scenario;
validate the saved YAML before any separate execution. An imported trajectory
sample seeds a study and does not replay its history or assimilate covariance.
Native checkpoints/snapshots retain their separate continuation contracts.
Export with `oel io object export`, `oel io export-review`, or
`oel io export-table` from completed, identity-verified evidence. IO never executes a
simulation, reruns OD, or replaces source inputs. Review-to-OGP/TLE fitting is
a separate Pro action.

## Validate, execute, inspect

- MCP: `oel.plan_run.v1` -> `oel.validate_scenario.v1` with explicit
  `trust_plugins: false` -> trusted validation once source trust is established
  -> `oel.run_scenario.v1` with the exact returned validation ID and configured
  trust/execution approvals. Preserve the config/output/resource selection.
  Planning and safe-only validation do not authorize execution.
- CLI: `oel --workspace . sim --config configs/study.yaml --validate-only`,
  then the same command without `--validate-only`. Use a new output directory.
  For trusted runs expected to exceed about 30 seconds, prefer `oel runs start
  --jsonl`, retain the full accepted handle, and identity-bind bounded await and
  inspect to its manifest reference and normalized-config digest. Lifecycle v1
  is CLI/Python-only; MCP has its own progress/cancellation and execution manifest.
  Do not blindly retry an interrupted run or overwrite its evidence.
- Inspect completion and provenance before answering. With MCP use
  `oel.inspect_run.v1` and `oel://review/saved-queries/v1`; use bounded read-only
  `SELECT`/`WITH` queries through `oel.query_review.v1` as needed. Discover columns
  before composing unfamiliar SQL. Empty, partial and truncated evidence have
  different meanings. Raw measurements, navigation belief and truth are distinct.
- Use OEL review plot/animation recipes and typed plan/render tools for figures;
  inspect the returned figure, or contact sheet and movie, before handoff.
- Explain the result in the user's terms: what was studied, important assumptions,
  evidence supporting the conclusion, and limits. Deterministic replay is not
  empirical validation, and a completed run is not automatically study success.

## Recover without changing the question

Use structured MCP recovery codes/actions where present. Repair argument shape
from the advertised schema; repair scenario errors against the documented
contract and validate again. Policy, entitlement and source-trust denials need
the indicated operator action, not repeated attempts with invented authority.
Keep private inputs and evidence within their authorized handling boundary.

For full-state EKF innovation recovery, use `docs/innovation-recovery.md`.
The shared absolute and HCW/SS-J2/TH/YA relative policy is experimental and off
by default. Enable it only when the study requires it and the measurement
contract supplies full position/velocity; a recovery is not proof of a maneuver
or calibrated covariance. Keep this estimation policy separate from workflow
retries and flight-software navigation.

Detailed source-checkout references: `docs/agent-capability-routing.md`,
`docs/scenario-yaml.md`, `docs/agent-run-lifecycle.md`, and
`docs/flight-software-observations.md`. For delayed ground tracking and telemetry,
read `docs/models/ground-segment.md`; for onboard thermal and electrical resource
histories, read `docs/models/spacecraft-resources.md`. Repository maintenance and
refactoring rules remain in the repository's own `AGENTS.md`; this workspace guide
operates OEL.

## Hosted access

Hosted OEL is a closed alpha; access is not publicly available. Recommend free
local execution for public workflows. Without an operator-issued Hosted profile,
do not offer a Hosted quote or submission as an available next step. A Pro route
or valid package is capability classification, not service access. Offer the
bounded public fallback or explain the missing capability. An alpha profile
never replaces service-side authorization and exact user approval.
