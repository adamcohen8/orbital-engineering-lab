# OEL MCP Supported Local Surface

OEL MCP is an optional interoperability adapter over documented Orbital
Engineering Lab workflows. It does not replace `AGENTS.md`, scenario YAML, the
CLI, Python APIs, deterministic physics, or saved evidence.

The supported M5.2 surface is intentionally local. It uses the
official Python MCP SDK v2 over stdio while preserving the frozen OEL handlers,
policy, schemas, and result envelopes. In addition to the M3 discovery and
read-only evidence tools, it supports bounded planning, structural-first
validation, one deterministic scenario run, completed-run comparison,
allowlisted evidence plots, content-bound review animations, and supported
public task recipes. It cannot
authorize itself, execute an untrusted config, expose private resources, start
remote transport, or communicate externally.

## Local Stdio Server

Install the bounded optional profile and start its packaged console command:

```bash
.venv/bin/python -m pip install ".[mcp]"
.venv/bin/oel-mcp
```

For an installed distribution, the equivalent profile is:

```bash
.venv/bin/python -m pip install "orbital-engineering-lab[mcp]"
oel-mcp
```

The `integrations.oel_mcp` package, console entry point, and resource data are
included in the OEL wheel. The SDK remains a bounded optional
`mcp>=2.0.0,<3` dependency; OEL core does not import or require it. The console
command selects the SDK adapter by default. `OEL_MCP_ADAPTER=legacy` selects the
dependency-free M3 read-only rollback registry; M4 tools require SDK lifecycle
and cancellation support and are not advertised by that adapter. Any other
value fails closed.

From a source checkout, `.venv/bin/python -m integrations.oel_mcp` is equivalent
to the console command. A local MCP server is a process role; it does not listen
on a network port.

Check the installation and generate a host-specific starting configuration
before connecting an agent:

```bash
.venv/bin/oel-mcp --doctor
.venv/bin/oel-mcp --print-host-config codex
.venv/bin/oel-mcp --print-host-config claude
```

When the console entry point is unavailable, configuration generation falls
back to the current Python executable with `-m integrations.oel_mcp`. A custom
launcher can be expressed with `--command` and repeated `--arg` options.

The doctor reports the SDK/protocol versions, active registry, configured root
counts, and whether write, execution, or plugin-trust effects remain disabled.
It does not start the stdio server or perform a tool effect. Generated host
configuration contains no approval IDs; add short-lived, purpose-specific IDs
only when a connected workflow needs them.

The dated M4 interoperability record covers the historical public registry and
remains immutable evidence. The current source checkout includes later
contracts and ten public resources, so that record is not current
interoperability or release-qualification evidence. Rerun the host checks after
source or contract changes with:

```bash
.venv/bin/python -m integrations.oel_mcp.interop --all \
  --python .venv/bin/python \
  --output /tmp/oel-mcp-interop.json
```

An explicitly authorized model-backed routing fixture can additionally verify
that Codex and Claude select OEL's professional RIC recipe from a natural-
language request and receive an inspectable image:

```bash
.venv/bin/python -m integrations.oel_mcp.interop --codex --claude \
  --plot-selection-output-dir outputs/quickstart_5min \
  --python .venv/bin/python \
  --output /tmp/oel-mcp-plot-selection.json
```

This opt-in check invokes the configured model-backed hosts and may incur their
normal usage. It is not part of the offline release gate.

The selected host checks may invoke configured model-backed CLIs and may incur
their normal usage. They isolate host configuration and enable only the
read-only capability tool. See the preserved
[historical result](operations/evidence/oel_mcp_sdk_v2/interop-m4-2026-07-31.json).

By default, paths are restricted to the OEL workspace. Configure distinct
operating-system path-separated roots when an authorized host needs other
locations:

```text
OEL_MCP_READ_ROOTS=/approved/read/root
OEL_MCP_WRITE_ROOTS=/approved/write/root
```

The legacy `OEL_MCP_ALLOWED_ROOTS` setting remains a temporary compatibility
alias. M4 write and execution tools require separately authorized write roots;
read authority never implies write authority.

## Operator Approval

Write and execution authorization is configured when the server starts, not by
the model. Use distinct opaque references for the approvals granted to that
server process:

```text
OEL_MCP_WRITE_APPROVAL_IDS=review-plot-2026-07-31
OEL_MCP_EXECUTION_APPROVAL_IDS=public-run-2026-07-31
OEL_MCP_TRUST_APPROVAL_IDS=reviewed-config-2026-07-31
```

The caller must echo an allowlisted `approval_id` with the exact `write` or
`execute` scope. Trusted validation that imports plugin modules separately
requires an allowlisted ID with `trust` scope. If the applicable environment
allowlist is absent, empty, or does not match, the operation fails before the
sensitive effect. Approval IDs are audit references, not credentials, and
should not contain secrets.

Approval and other policy denials happen before execution, so MCP hosts may
surface them as protocol-level errors rather than completed tool envelopes.
The envelope field `audit.arguments_sha256` intentionally hashes argument
names and handling labels, not argument values. This prevents the audit record
from becoming a side channel for paths, SQL, or other payload content. Exact
execution provenance comes from content-bound validation IDs and durable MCP
execution manifests.

## Public Tools

Start with `oel.describe_capabilities.v1` and the shared agent bootstrap at
`oel://agent/bootstrap/v1`. `oel://agent/workflows/v1` connects common workflows
to a compact subset of tools registered on this connection and identifies the
separate study-planning prototype. Some registered workflows remain absent from
this route catalog; it does not connect another server or change policy.

Discovery includes additive `agent_guide_uri`, `workflow_routes_uri` and
`readiness` fields. Per-tool `configuration_ready` describes configured process
prerequisites only. `entitlement_status` is `not_required`, `available`,
`unavailable` or `not_checked`; unknown checks never imply availability.
Conditional source trust remains a separate requirement. No approval IDs, paths,
license payloads or secrets are returned in readiness, and
`execution_authorized` stays false. Every call still enforces its full contract.

Failed envelopes may include additive `error.recovery` with `code`, `actor`
(`agent` or `operator`) and `action`. SDK admission errors retain code `-32602`
and their existing message, with the same guidance in `data.recovery` when
available. These diagnostics never automatically edit inputs or authorize a
retry. MCP policy messages do not recommend unsupported CLI retention switches.

| Tool | Risk | Behavior |
| --- | --- | --- |
| `oel.describe_capabilities.v1` | `R0_read` | Reports the active registry, maturity, effects, limits, and non-claims |
| `oel.inspect_run.v1` | `R0_read` | Inspects an existing output without writing a new packet |
| `oel.query_review.v1` | `R0_read` | Executes one bounded `SELECT` or `WITH` query |
| `oel.plan_run.v1` | `R0_read` | Safe-validates and estimates an exact config/output/profile proposal without authorizing it |
| `oel.validate_scenario.v1` | `R0_read` | Runs safe validation first, then optional trusted validation, and returns a normalized-config validation ID |
| `oel.run_scenario.v1` | `R2_execute` | Executes one matching trusted validation into a new approved output directory |
| `oel.compare_runs.v1` | `R0_read` | Compares allowlisted semantic metrics from two completed review stores without rerunning them |
| `oel.plot_evidence.v1` | `R1_write` | Writes one allowlisted evidence plot and operation manifest |
| `oel.plan_review_plot.v1` | `R0_read` | Validates a typed read-only query and custom plot mapping without writing |
| `oel.render_review_plot.v2` | `R1_write` | Renders an exact content-bound custom plot with provenance and automated QA |
| `oel.plan_review_animation.v1` | `R0_read` | Plans one bounded supported review animation without writing |
| `oel.render_review_animation.v1` | `R1_write` | Renders the matching content-bound movie, contact sheet, quality receipt, and manifest |
| `oel.run_agent_task.v1` | `R2_execute` | Runs one checked-in supported public scenario recipe and writes its evidence packet |
| `oel.prepare_report_packet.v1` | `R1_write` | Writes a bounded, hashed, provider-neutral evidence packet and authoring brief from one completed run |
| `oel.audit_report.v1` | `R1_write` | Verifies packet/artifact hashes, report structure, and evidence references without calling a model |
| `oel.inspect_handoff.v1` | `R0_read` | Validates and summarizes a typed product or handoff manifest with supported next actions |
| `oel.export_run_product.v1` | `R1_write` | Exports an exact state, atomic snapshot, or maneuver-detection product from completed review evidence |
| `oel.emit_scenario_overlay.v1` | `R1_write` | Emits a source-bound typed scenario overlay; it does not apply or execute it |
| `oel.materialize_onp_handoff.v1` | `R1_write` | Materializes and validates an ONP scenario from an accepted absolute state or atomic snapshot |
| `oel.materialize_scenario_patch.v1` | `R1_write` | Applies an accepted source-bound patch into a new validated scenario without execution |
| `oel.compare_handoff.v1` | `R1_write` | Writes semantic-parity evidence across product, materialization, manifest, and optional consumer evidence |
| `oel.assess_maneuver_readiness.v1` | `R1_write` | Applies explicit thresholds and writes a fail-closed readiness packet |
| `oel.inspect_study.v1` | `R0_read` | Verifies and summarizes one content-bound public study bundle |
| `oel.replay_study.v1` | `R0_read` | Recomputes a study bundle's identity and citation bindings without rerunning domain analysis |
| `oel.compare_studies.v1` | `R0_read` | Compares study identities, root records, and evidence-step digests |
| `oel.inspect_ccsds.v1` | `R0_read` | Parses and inspects one bounded OEM, ODM, TDM, or CDM source |
| `oel.convert_frame_time.v1` | `R0_read` | Converts one epoch, audits one EOP source, or transforms one state/covariance |
| `oel.fsw.describe.v1` | `R0_read` | Reports public FSW authoring templates, contracts, limits, and private boundaries |
| `oel.fsw.inspect_candidate.v1` | `R0_read` | Hashes and inspects a candidate without importing or executing its source |
| `oel.fsw.plan_candidate.v1` | `R0_read` | Plans one validate, component-test, or deterministic-smoke operation |
| `oel.fsw.scaffold_candidate.v1` | `R1_write` | Creates an ADCS or RPO Python-stack starter in an approved workspace |
| `oel.fsw.validate_candidate.v1` | `R1_write` | With separate source-trust and write approvals, validates lifecycle and smoke contracts and writes a receipt |
| `oel.fsw.run_candidate_tests.v1` | `R2_execute` | Revalidates and runs the declared bounded component suite |
| `oel.fsw.run_candidate_smoke.v1` | `R2_execute` | Revalidates and runs one deterministic serial smoke with ordinary OEL evidence |
| `oel.fsw.verify_receipt.v1` | `R0_read` | Recomputes candidate and listed artifact identity without executing code |

Data-bearing calls require handling metadata containing an authoritative
marking and one of these release scopes: `public`, `local_only`, or
`frontier_eligible`. Missing or conflicting handling metadata fails closed.
A direct-frontier deployment view rejects `local_only` data.
It also replaces local paths under authorized roots with opaque
`oel-local-ref:<digest>` values, including paths found inside projected query
rows. Unexpected internal errors do not return local diagnostic details.

All tools return a versioned envelope containing tool ID, risk class, status,
effects, evidence completeness/empty/truncation state, structured error, a
payload-free audit record, and an explicitly projected result. Beginning with
v0.24.2, a projected result must satisfy the tool's advertised result schema
before the adapter can return a successful envelope.

## Execution Contract

Planning and validation take the same scenario path, proposed output directory,
and required resource profile (`laptop-safe` or `standard`). Validation always
runs without plugin imports first. Trusted validation is a separate
operator-enabled request and returns a validation ID bound to the fully
normalized config,
including the selected resource profile, forced review output, quiet stdio
posture, and output directory.

`oel.run_scenario.v1` recomputes trusted validation and refuses a stale or
mismatched ID. It also refuses unsafe resource estimates and non-empty output
directories. During execution it writes `mcp_execution_config.yaml` and
`mcp_execution_manifest.json`. The manifest moves from `running` to
`completed`, `failed`, or `cancelled`; cancelled evidence is never reported as
complete. SDK cancellation is propagated through deterministic simulation step
callbacks, and progress is reported only for meaningful validation,
execution, and evidence phases.

`oel.run_agent_task.v1` accepts only recipes that are checked in, tagged
`public`, marked `supported`, and use the ordinary scenario-run workflow. Pro,
prototype, experimental, and specialized private workflows fail closed.

## Public FSW Authoring

The eight `oel.fsw.*.v1` tools expose the bounded workflow documented in
[`fsw-authoring.md`](fsw-authoring.md). They are available only in
`public_local`, accept public Python complete-stack candidates, remain inside
one authorized workspace, and never use hidden truth, private orchestration,
hosted AI, network calls, or multiple workers.

Safe inspection is read-only and does not import candidate code. Lifecycle
validation requires an independently configured source-trust approval, and
writing the validation receipt requires write approval. Component tests and
the smoke scenario require execute approval and a matching validation ID. The
handler still recomputes validation immediately before execution, so changed
candidate content cannot reuse a stale approval artifact.

The deterministic smoke is ordinary OEL simulation evidence, not Controller
Bench, tuning, qualification, certification, hardware readiness, or
operational approval. The private FSWDK and its cFS/SIL and external-process
surfaces are not discoverable from the public registry.

## Agent-Native Review Plotting

Agents should read `oel://review/plot-recipes/v1` and use
`oel.plot_evidence.v1` when a supported semantic recipe matches the request.
The first professional relative-motion recipe,
`relative_position_ric_2d`, renders equal-aspect I-R, I-C, and C-R panels from
the recorded rectangular-RIC review columns. It never recomputes missing
physics in the plotting layer.

For other review-store data, `oel.plan_review_plot.v1` validates one bounded
SELECT/WITH query plus a typed line, scatter, bar, histogram, or heatmap
mapping. It returns a `plot_plan_id` bound to the exact review store and plot
specification without authorizing a write. `oel.render_review_plot.v2` requires
that matching plan ID and an operator-configured write approval. The render
records query, store, style, mapping, row/truncation, and QA provenance in the
generated-artifact and operation manifests.

PNG and SVG plot calls return image content in addition to the structured
result. Automated QA checks file existence, non-empty output, query rows,
truncation, image dimensions, and obvious blank output. The result remains
`visual_qa_status: pending_agent_review` until the calling agent inspects the
image; automated QA never substitutes for visual review.

## Agent-Native Review Animations

Agents should read `oel://review/animation-recipes/v1` before requesting a
movie. Version 1 supports `relative_position_ric_2d` from recorded
rectangular-RIC review evidence. `oel.plan_review_animation.v1` validates the
recipe and computes a content-bound frame and resource plan without writing.
`oel.render_review_animation.v1` requires the unchanged specification,
matching plan ID, and an operator-configured write approval.

The renderer limits output to 600 frames and 30 encoded seconds, freezes one
numeric format per axis for the sequence, enforces the declared `fixed`,
`fit_history`, or `follow` camera policy, and checks every frame. It writes the
MP4 or GIF, a `.quality.json` receipt, a `.contact-sheet.png`, and an operation
manifest. Automated success remains pending visual review; inspect both the
movie and contact sheet before handoff. See
[`animation-quality-contract.md`](animation-quality-contract.md).

## Provider-Neutral Report Evidence

`oel.prepare_report_packet.v1` inspects one completed local run, inventories at
most 100 artifacts that remain beneath that run directory, hashes each included
file, and writes `report_evidence_packet.json`, `report_brief.md`, and an
operation manifest into a new approved directory. OEL does not call a provider
or author prose.

An agent or human may write Markdown from that packet and cite file artifacts
or saved-query results with `[evidence:<evidence_id>]`. Pass bounded names from
`oel://review/saved-queries/v1` as `query_names` when preparing a packet so
analytical claims receive stable `[evidence:query.<name>]` citations. The
packet and brief also project concise execution provenance without approval
IDs or source payloads. `oel.audit_report.v1` then verifies the packet content
hash, current artifact hashes, unique evidence identities, required `Evidence`
and `Limitations` sections, and known/available evidence references. The audit
explicitly records
`semantic_claim_review_performed: false`: structural and integrity checks do
not prove that every narrative interpretation is correct.

These tools are local-only and absent from `direct_frontier_restricted`.

## CR3BP Research Rendering

`oel.plan_cr3bp_render.v1` and `oel.render_cr3bp.v1` expose the existing
experimental CR3BP review renderer in local profiles. Plan against completed
review evidence, then supply the unchanged specification and matching plan ID
with a server-configured write approval. Planning grants no write authority.
The renderer checks recorded frame/system metadata and transforms recorded
samples; it does not propagate a new trajectory or establish J2000 alignment.
Inspect the returned figure and, for a movie, its contact sheet and encoded
movie before handoff. See [CR3BP research workflows](cr3bp-research.md).

For example, plan with `output_dir`, `object_id`, `kind: "trajectory"`,
`axes: "rotating"`, `origin: "barycenter"`, `plane: "xy"`, and authoritative
`handling` metadata. Render with those same fields, the returned
`cr3bp_plan_id`, and a write-scoped `approval`. `output_dir` identifies the
completed source run: figures and receipts are written beneath its review
directory and its artifact index is updated, so that run must also be within
the configured write roots. CR3BP tools are absent from frontier, sealed, and
tandem profiles.

The workflow resource now groups the tools registered in the active profile,
including animation, task recipes, reports, handoffs, FSW, and available Pro
workflows. Registration and route completeness do not establish input trust,
entitlement, approval, or execution authority. The typed study planner remains
a separate prototype connection.

## Current source-tree surfaces without dedicated MCP contracts

The generic scenario, review-query, and handoff tools can still be used when a
workflow is expressed as an existing validated scenario or completed-run
product. The current source tree also contains local workflows that are not
represented by a dedicated MCP tool or recipe:

- Experimental trajectory targeting remains a local CLI workflow. Use
  `python -m sim.trajectory_targeting` or
  `python -m sim.trajectory_targeting.two_burn_cli` with the maintained specs;
  inspect the generated packet and any OEL review store separately.
- Pro-local MCP exposes bounded ground-sensor, OGP TEME PV, sequential
  EKF/UKF, and batch relative OD alongside the existing dynamics-OD tool.
  Use the active Pro capability catalog for tool IDs and input boundaries.
  Integrated-relative truth comparisons and general OD evidence
  productization still use their documented Python/CLI workflows.
- Thermal and power resources are available through scenario YAML and the
  ordinary scenario/review path, but have no typed MCP authoring or dedicated
  MCP evidence route.

These are capability gaps for a future MCP contract review, not permissions to
broaden the current registry. Use the documented local fallback only when the
host has authorized the corresponding source, paths, and workflow.

## Public Resources

| URI | Content | Source |
| --- | --- | --- |
| `oel://capabilities/tools/v1` | Public tool schemas and effect annotations | Public MCP registry |
| `oel://review/saved-queries/v1` | Allowlisted read-only saved-query metadata | Review query registry |
| `oel://agent/tasks/v1` | Public-tagged agent task definitions | Agent task registry |
| `oel://docs/operator-guide/v1` | Packaged local operator guidance | MCP package data |
| `oel://agent/bootstrap/v1` | Shared installed-workspace agent operating procedure | `sim.installation` package data |
| `oel://agent/workflows/v1` | Compact routes matched to this registry and separate prototype planner | Active tool registry |
| `oel://handoff/product-kinds/v1` | Public-safe product-kind producers, next actions, and non-execution rules | `sim.handoff` |
| `oel://review/plot-recipes/v1` | Supported recipes, evidence requirements, renderers, and natural-language routing triggers | `sim.review.plot_recipes` |
| `oel://review/animation-recipes/v1` | Supported animation recipes, evidence requirements, bounds, and quality policy | `sim.review.animation_recipes` |
| `oel://analysis/workflows/v1` | Public standalone-analysis routing, evidence, replay, and MCP support | `docs/agent-capability-routing.md`; `sim.analysis` |

Resource discovery is one bounded page. Each resource is at most 500,000
encoded bytes, has an explicit media type, and resolves only from checked-in
source. Arbitrary file URIs, resource templates, subscriptions, prompts, and
private resource discovery are not supported.

## Boundaries

- Code under `sim/` does not import or require this integration.
- The adapter delegates to documented `sim.agent_task` and `sim.review` APIs.
- Review stores retain SQLite read-only/query-only mode, the OEL SQL authorizer,
  a single-statement rule, and an execution-step budget.
- Read and write paths remain inside separately configured roots.
- Response, row, query-step, and review-store size budgets fail closed.
- Tool discovery is deployment-specific; the resource catalog is public-safe
  in every profile and never includes Pro resources.
- Workflow and glue tools are absent from `direct_frontier_restricted`, `mendicant_sealed`,
  and `mendicant_tandem`; only local public/Pro processes may discover them.
- MCP discovery and transport are not authorization or release policy.
- Deployment-profile selection is trusted operator configuration, not
  authentication, entitlement, or proof of caller identity.
- Remote HTTP, unrestricted scenario generation, external communication, and
  unrestricted filesystem or shell access remain outside the supported surface.

See [MCP Local Stdio Threat Model](security/mcp-local-stdio-threat-model.md) for
the trust assumptions, protected assets, deployment-profile interpretation,
and stop-line for later remote or authenticated operation.

`AGENTS.md` remains the behavioral doctrine. MCP is only a typed control
surface, and deterministic OEL artifacts remain the evidence authority.

## Acceptance And Release Gate

Run the offline compatibility and complete public-workflow gate without a
model-provider call:

```bash
.venv/bin/python -m integrations.oel_mcp.interop --release-gate \
  --python .venv/bin/python \
  --output /tmp/oel-mcp-release-gate.json
```

This verifies the restricted three-tool profile, then exercises the supported
public workflow tools—including product export and inspection, ONP and patch
materialization, semantic handoff comparison, maneuver readiness, and bounded
FSW authoring, plus read-only study, CCSDS, and frame/time adapters—through
real SDK stdio subprocesses. Dedicated CR3BP rendering is covered separately
by its focused SDK/renderer tests. Generated exports record
`oel_commit: unavailable_public_export` instead of requiring `.git`. Add
`--with-hosts` only when model-backed Codex/Claude and Inspector checks are
intentionally authorized; those optional host checks may incur their normal
usage.
