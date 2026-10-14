# OEL MCP Operator Guide

This packaged guide covers the supported local stdio OEL MCP surface.

Agents should start with `oel.describe_capabilities.v1`, then read
`oel://agent/bootstrap/v1` and `oel://agent/workflows/v1`. The bootstrap is also
installed as `AGENTS.md` in new managed workspaces. Workflow routes provide a
compact subset of tools registered here and distinguish them from the separate
`oel-study-mcp` planning prototype; some registered workflows remain absent
from the route catalog. Configure that connection through the host operator
only when needed.

Discovery's `readiness.tools` reports process-configuration blockers and local
entitlement status without exposing approval IDs or granting authority. A
`configuration_ready` tool still checks input handling, exact approval IDs,
trust, paths, validation and resource bounds on every call. For errors, use
optional `error.recovery` in tool envelopes or `data.recovery` in SDK admission
errors. These contain a stable code, the actor needed and an applicable action.

- Install with `python -m pip install "orbital-engineering-lab[mcp]"`.
- Start with `oel-mcp`; the official SDK adapter is the default.
- Run `oel-mcp --doctor` before connecting a host. Use
  `oel-mcp --print-host-config codex` or `claude` for a launchable starting
  configuration. For a custom command, repeat `--arg` for every server
  argument.
- Use `OEL_MCP_ADAPTER=legacy` only as the documented M3 read-only rollback
  path; it does not advertise M4 workflow tools.
- Configure the narrowest required `OEL_MCP_READ_ROOTS` and
  `OEL_MCP_WRITE_ROOTS`.
- Configure `OEL_MCP_WRITE_APPROVAL_IDS` and
  `OEL_MCP_EXECUTION_APPROVAL_IDS` outside the model before enabling M4
  write or execution calls. Configure `OEL_MCP_TRUST_APPROVAL_IDS` before a
  validation call may import scenario plugins. Approval IDs are non-secret
  audit references.
- Treat deployment profiles and handling metadata as operator policy inputs,
  not authentication or release approval.
- Treat deterministic OEL artifacts and review stores as the evidence
  authority.
- Read `oel://review/saved-queries/v1` before composing free-form SQL. Pass the
  relevant saved `query_names` to `inspect_run` or `prepare_report_packet` so
  analytical claims can cite stable query evidence IDs instead of only the
  entire review store.
- Coverage studies can request `coverage_summary` and
  `coverage_transition_summary`; directed-link studies can request
  `directed_link_summary` and `directed_link_windows`. These queries require
  the corresponding coverage or link tables in the completed review store;
  their presence is not inferred and they may legitimately return no rows.
- Read `oel://review/plot-recipes/v1` before plotting review evidence. Use a
  supported OEL recipe when one matches; otherwise validate the exact query and
  mapping with `oel.plan_review_plot.v1` before the approved
  `oel.render_review_plot.v2` write. Inspect the returned image before handoff.
- Read `oel://review/animation-recipes/v1` before animating review evidence.
  Plan with `oel.plan_review_animation.v1`, then render the matching content-
  bound plan with `oel.render_review_animation.v1` and an approved write ID.
  Inspect both the contact sheet and encoded movie before handoff.
- For GNC v2 evidence, prefer `fsw_invocation_summary`,
  `fsw_sensor_deliveries`, `actuator_command_chain`, `fsw_deadline_misses`,
  `safety_requirement_status`, and `fsw_checkpoint_summary`. These are
  additive saved-query names; MCP tool versions and approval semantics are
  unchanged.
- Read `oel://handoff/product-kinds/v1` before routing a typed product. Inspect
  a product before consuming it; materialization tools validate and write new
  scenarios but never authorize or execute those scenarios.
- Read `oel://analysis/workflows/v1` before routing a standalone orbital
  analysis. The resource distinguishes scenario YAML from versioned analysis
  problems and lists the supported evidence/replay path. Only study, CCSDS, and
  frame/time inspection/conversion are exposed as public OA MCP adapters; use
  the documented CLI or Python API for the other workflows.
- `oel.inspect_study.v1`, `oel.replay_study.v1`, and
  `oel.compare_studies.v1` verify retained study identity and citations; they
  do not rerun domain physics. `oel.inspect_ccsds.v1` and
  `oel.convert_frame_time.v1` return bounded receipts and never execute a
  scenario.
- Policy and approval denials occur before tool execution and may be returned
  by a host as protocol-level MCP errors. Operation failures after policy
  admission use the standard structured OEL envelope.
- `audit.arguments_sha256` deliberately hashes argument names and handling
  labels only. It is a payload-free shape audit, not a unique invocation or
  evidence identity; use validation IDs and execution manifests for content
  provenance.

The local surface also supports content-bound planning/validation, one approved
deterministic scenario run, completed-run comparison, allowlisted plots, and
supported public scenario-task recipes. Validation never authorizes execution.
Run tools require a matching trusted validation ID where applicable, a safe
resource preflight, a new output directory, and a server-configured approval.

The ordinary scenario tools can run a supported existing config that opts into
the headless `ground_segment` layer or ONP `specs.thermal`/`specs.power` models;
these are scenario features, not standalone analysis tools. Use the
`docs/agent-capability-routing.md` entry points and the corresponding
`docs/models/ground-segment.md` or `docs/models/spacecraft-resources.md` contract
to prepare and interpret those runs. There is no separate MCP tool that authors
these scenario sections.

Provider-neutral report packet and audit tools can hash completed local
evidence and verify report structure/references. They do not call a model or
perform semantic claim review.

The glue-aware surface can export exact completed-run states, atomic snapshots,
and maneuver detections; materialize accepted products or patches; compare
handoff semantics; and assess maneuver readiness.
Every write/execute effect requires an operator-configured approval, uses a new
output target, and records a durable operation manifest.

Successful results are checked against the advertised result schema. Pro Scale
refresh runs against an isolated snapshot of the exact validated store, and
Pro workflows that require more than one feature report and enforce every
entitlement.

Discover active workflow groups with `oel://agent/workflows/v1`. Local CR3BP
research figures and movies use `oel.plan_cr3bp_render.v1`, followed by
`oel.render_cr3bp.v1` with an unchanged specification, matching
`cr3bp_plan_id`, and configured write approval. The source run must be writable
because its review artifacts and index are updated. These experimental
transforms do not establish J2000 alignment or rerun dynamics; inspect the
image and any contact sheet/movie before handoff.

Entitled Pro-local profiles also expose bounded ground-sensor, OGP TEME PV,
sequential EKF/UKF, and batch relative OD with JSON inspection. Inspect the
active tool schemas and readiness first. Execution requires local-only
handling and configured execution approval; derived products remain subject
to their quality gates and do not carry execution authority.

Trajectory targeting still uses `python -m sim.trajectory_targeting ...`.
Integrated-relative truth comparisons and general OD evidence packaging use
the documented `sim.estimation` Python/CLI surfaces. Thermal and power
resources use the generic scenario/review path; MCP has no dedicated typed
authoring route for them. Use local fallbacks only within the host-authorized
workflow and paths.

Remote transport, arbitrary filesystem resources, prompts, sampling,
unrestricted scenario generation, private campaigns, and frontier release
brokering are not part of the supported local surface.
