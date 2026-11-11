# Agent Task Runner

Commands in this guide use `python` after the OEL virtual environment has been
activated. See [Installing OEL](installation.md) for explicit Windows
PowerShell and macOS/Linux paths.

`sim.agent_task` is an optional orchestration layer for OEL-capable agents. It
packages common validate/run/inspect workflows into machine-readable recipes
and writes an `agent_evidence_packet.json` that can be cited, diffed, or handed
to another tool.

It does not replace scenario YAML, `run_simulation.py`, the deterministic
simulator, or `sim.review`. The runner prepares normal scenario YAML with
`outputs.review.enabled: true`, validates it through the public Python API,
runs it through the canonical execution service, queries the review store, and
saves the evidence it used.

## When To Use It

Use the task runner when an agent needs a repeatable, evidence-backed workflow:

- running a known public recipe and capturing the queries used;
- inspecting an existing output directory without ad hoc JSON parsing;
- comparing two scenario configs through the same review metrics;
- generating a standard plot from a completed review store.

For one-off scenario authoring, agents should still design the smallest
scenario that answers the user's question, validate it, run it, then inspect
the generated evidence. The task runner is a shortcut for common inspection and
handoff patterns, not a reason to force a user request into a canned example.

Bundled recipes carry a machine-readable maturity label:

| Maturity | Meaning |
| --- | --- |
| `supported` | A maintained product workflow with documented commands, evidence packet shape, and regression coverage. Agents may use it as a default path when it matches the user's actual task. |
| `prototype` | A useful repeatable workflow or smoke check whose outputs need analyst interpretation before being treated as product evidence. |
| `experimental` | A discoverable recipe for exploration only; do not present it as a supported analysis workflow. |

Public-tagged bundled recipes must be `supported`. Private-service recipes may be
`supported`, `prototype`, or `experimental`, but the answer should name that
maturity when the evidence depends on the recipe.

Recipe maturity describes orchestration, query, plot, and packet stability; it
does not promote the underlying scientific/product claim. For example,
`coverage_link_review` is a supported wrapper around an experimental,
evidence-only coverage/RF analysis. Answers must carry that domain support
level and its calibration, weather, interference, scheduling, and assurance
non-claims.

## Commands

List bundled recipes:

```bash
python -m sim.agent_task list
python -m sim.agent_task list --json
```


List named plot recipes:

```bash
python -m sim.agent_task list --plots
```

Inspect semantic metric definitions:

```bash
python -m sim.agent_task semantics
python -m sim.agent_task semantics closest_approach_km --json
```

Run a recipe and write an evidence packet:

```bash
python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --json
```

`plot_recipe_ids` advertises available follow-up figures; recipes do not render
them by default. Pass the machine-readable `plot_cli_option` (currently
`--plot`) when the task needs those artifacts:

```bash
python -m sim.agent_task run attitude_hold_review \
  --output-root outputs/agent_tasks --plot --json
```

Validate a recipe without executing it:

```bash
python -m sim.agent_task run quickstart_review --output-root outputs/agent_tasks --dry-run
```

Inspect a completed output directory:

```bash
python -m sim.agent_task inspect outputs/quickstart_5min \
  --query run_metadata \
  --query rendezvous_closest_approach \
  --json
```

When inspecting an existing directory, treat the packet as evidence for that
directory's current contents, not proof that the source scenario still produces
the same result. If a local output may be stale, rerun the recipe or scenario
before citing metrics.

Default inspection and MCP report packets automatically add the named Coverage
and Directed-Link summary/window queries when those domain tables contain
evidence. Runs without those analyses retain the compact metadata/object/artifact
default instead of receiving unexpected-empty domain queries.

Compare two configs through common review metrics:

```bash
python -m sim.agent_task compare \
  --base configs/quickstart_5min.yaml \
  --candidate configs/ric_pd_10km_experiment.yaml \
  --output-dir outputs/agent_tasks/quickstart_vs_flagship \
  --metric closest_approach_km \
  --metric final_range_km \
  --json
```

Comparison packets include the requested metric names, extracted values for
each side, numeric deltas where both values are available, and a
`metric_status` row for each requested metric. Check `metric_status` before
citing a delta; it records whether the base and candidate evidence both
contained the metric and whether the metric resolved to a known semantic
definition.

Create a standard plot from a completed review store:

```bash
python -m sim.agent_task plot outputs/quickstart_5min \
  --recipe relative_range \
  --style oel_light \
  --format png
```

## Evidence Packet

Each run, inspect, or compare command writes:

```text
<output-dir>/agent_evidence_packet.json
```

The packet includes:

- task id, status, generated UTC timestamp, and packet schema version;
- top-level evidence summary status;
- recipe metadata when a bundled recipe was used;
- prepared scenario config paths and output directories;
- validation reports;
- run summaries when execution occurred;
- review-store table/schema metadata;
- saved review query summary status;
- saved review query request status, maturity, source tables, SQL, columns,
  row samples, empty-result policy, empty-result status, and truncation status;
- requested semantic metric names, including unknown-name status;
- resolved semantic metric definitions used by the task;
- generated artifact summary status;
- generated plot/artifact paths and artifact path-existence status;
- generated plot summary status;
- structured failure hints when validation, review queries, or artifact
  inspection fail.

Agents should cite the packet path and the query name or SQL used when a claim
depends on run evidence.

Packets include `evidence_summary`, a top-level rollup of validation,
review-query, artifact, plot, failure-hint, and caveat status. Treat
`evidence_summary.ready_to_cite: false` as a cue to inspect the component
summary fields before relying on the packet.

Review blocks include `query_summary`, which counts successful, failed,
unknown, unexpected-empty, and truncated query rows. Treat
`query_summary.evidence_complete: false` as a signal to inspect the named query
lists before relying on the packet's conclusions. Failed, unknown,
unexpected-empty, or truncated query rows make the summary incomplete.

Saved query rows include `known`. Unknown saved query names are marked
`known: false`, `status: unknown_query`, and `reason: unknown_saved_query`.
Rows for known queries include `empty_result`, `empty_result_allowed`, and
`empty_result_unexpected`. Treat `empty_result_unexpected: true` as a weak or
missing evidence signal even when the SQL executed successfully.

Artifact sections include `artifact_summary`, which counts existing, missing,
and unknown-status artifact paths. Treat `artifact_summary.artifacts_complete:
false` as a signal to inspect `missing_artifacts` and
`path_status_unknown_artifacts` before citing generated files.

Plot sections include `plot_summary`, which counts successful, failed, missing,
and truncated generated plots. Treat `plot_summary.plots_complete: false` as a
signal to inspect `failed_plots`, `missing_plots`, and `truncated_plots` before
citing figures.

## Semantic Metrics

Semantic metrics are named review-store quantities that agents may cite in
answers, comparisons, and standard plot provenance. Each definition carries:

- a stable metric name, description, units, and interpretation guidance;
- a maturity label using the same `supported`, `prototype`, or `experimental`
  vocabulary as task and plot recipes;
- the review source tables that support the metric;
- an optional saved review query and read-only SQL snippet;
- caveats that must travel with the metric when they affect interpretation.

Supported semantic metrics must resolve to registered saved queries when they
name one, and any SQL definition must be `SELECT`/`WITH` only. The evidence
packet records the semantic metric definitions used by a task so downstream
agents can distinguish an observed number from the review evidence contract
that made it citeable.

Packets also include `semantic_metric_requests`, a request audit trail that
lists every requested metric name once. Known rows include maturity, source
tables, and saved query; unknown rows are marked `known: false` with
`reason: unknown_semantic_metric` so a typo or unsupported metric is visible
instead of silently disappearing from `semantic_metrics`.

## Bundled Recipes

Current bundled task recipes include:

- `quickstart_review`: public quickstart scenario with rendezvous metrics,
  closest approach, burn activity, and artifact inventory. Maturity:
  `supported`.
- `flagship_ric_pd_review`: flagship 10 km RIC PD scenario with terminal
  relative-state and burn evidence. Maturity: `supported`.
- `mission_reconstitution_review`: public mission-reconstitution trade-space
  example with recovery summary, candidates, and burn sequence evidence.
  Maturity: `supported`.
- `ground_access_review`: public ground-access example with station/object
  access summaries and no-access reason counts. Maturity: `supported`.
- `coverage_link_review`: public whole-Earth coverage and directed-link example
  with named summary/window queries and standard coverage-fraction and
  link-margin plots. Maturity: `supported`.
- `ogp_sgp4_review`: fixed public TLE propagated continuously through passive
  OGP-SGP4 with propagation/frame provenance, final canonical ECI state, and a
  review-store position plot. Maturity: `supported`.
- `ogp_sdp4_review`: fixed synthetic GEO-like TLE with valid checksums propagated
  continuously through passive OGP-SDP4, including resolved deep-space regime,
  period, frame, final-state, extrema, and plot evidence. Maturity: `supported`.
- `attitude_hold_review`: disturbed ideal-wrench attitude-hold example with
  desired/actual quaternion error, body rates, controller evidence, and supported
  review plots. Maturity: `supported`.

## Plot Recipes

Current bundled plot recipes include:

- `object_eci_radius`: maturity `supported`, table `object_state`
- `relative_range`: maturity `supported`, table `relative_state`
- `relative_range_rate`: maturity `supported`, table `relative_state`
- `relative_velocity_components`: maturity `supported`, table `relative_state`
- `relative_position_ric_2d`: maturity `supported`, table `relative_state`, professional I-R/I-C/C-R renderer
- `burn_activity`: maturity `supported`, table `thrust`
- `ground_access`: maturity `supported`, table `ground_access`
- `attitude_error`: maturity `supported`, table `attitude_error`
- `attitude_body_rates`: maturity `supported`, table `object_state`
- `coverage_fraction`: maturity `supported`, table `coverage_samples`
- `directed_link_margin`: maturity `supported`, table `link_samples`, specialized closure-threshold renderer

Plot recipes run read-only review SQL and call the review plotting service.
Generated plot provenance is recorded in `review/generated_artifacts.json`.
Agent-task plot artifacts also include the recipe maturity, source tables, and
semantic metric names so figures can be cited with their evidence contract.

## Failure Hints

The runner emits structured hints for common agent repair loops, including:

- plugin validation failures;
- missing review stores;
- review schema/query mismatches;
- YAML loading errors;
- invalid timing configuration;
- absent `review/run.sqlite` after execution.

Hints are guidance for the next deterministic repair step. They are not
simulation evidence by themselves.
