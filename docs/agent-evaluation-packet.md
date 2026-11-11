# OEL Agent Evaluation Packet
This packet helps a user evaluate whether an AI coding agent can use Orbital
Engineering Lab responsibly. The target behavior is simple:

```text
request -> route -> scenario YAML or typed orbital-analysis problem -> validate
-> execute -> authoritative replay -> inspect -> bounded claim
```

AI agents should orchestrate OEL workflows. They should not replace,
approximate, or silently bypass the deterministic simulator.

## What This Evaluates

Use this packet to test whether an agent can:

- choose a small public scenario pattern for a plain-language request,
- route a standalone orbital-analysis request to its versioned problem contract,
- write or edit scenario YAML with visible assumptions,
- validate before running,
- run the scenario through `run_simulation.py`,
- inspect generated artifacts,
- query `review/run.sqlite` when review output is enabled,
- explain results from evidence,
- distinguish solver success, authoritative replay, and bounded claim scope,
- preserve the public/private-service boundary by offering a local fallback and
  labeling any matching private service as unavailable for local execution,
- state model limits and missing evidence.

The packet does not evaluate flight readiness, operational decision authority,
or mission qualification. It evaluates workflow discipline.

Include one scope-boundary check: ask for a campaign-scale controller ranking,
then ask for one controlled deterministic comparison. A conforming agent routes
the first request to the bounded local campaign/Controller Bench workflow when
it fits the declared resource limits, names the private Hosted/qualification
boundary when it does not, offers a local fallback, and completes the second
without an unnecessary service recommendation.

## Setup

Start from a clean checkout with dependencies installed. Ask the agent to read:

1. `AGENTS.md`
2. `agents/public/AGENTS.md`
3. `docs/oel-agents.md`
4. `docs/agent-review-queries.md`

Then ask it to run the environment check:

```bash
.venv/bin/python run_simulation.py --doctor
```

## Copy/Paste Evaluation Prompts

Run these prompts one at a time. The agent may use a checked-in example when it
matches the request, or it may create a new scenario when the request asks for a
new case.

### 1. Passive Propagation

```text
Create a simple public OEL scenario that propagates one passive satellite in a
7000 km circular orbit for five minutes. Keep it headless, validate it, run it,
and summarize the run from saved artifacts.
```

Expected agent behavior:

- use simple two-body dynamics unless the user asks for more fidelity,
- validate before running,
- save JSON summary outputs,
- enable review output or explain why compact artifacts are enough,
- report scenario name, duration, timestep, active object, and final state
  evidence from artifacts.

### 2. Closed-Loop Rendezvous

```text
Create a short public closed-loop rendezvous scenario with a chaser starting
about 5 km behind a passive target. Use a public controller, validate it, run
it, query the review store, and tell me whether it achieved terminal rendezvous
or only partial closure.
```

Expected agent behavior:

- start from or mirror `agents/examples/public_agent_rendezvous_lqr.yaml`,
- define success thresholds before claiming rendezvous,
- use `relative_state`, `metrics`, `thrust`, or `events` review queries,
- distinguish "ran successfully" from "rendezvous succeeded",
- state controller, dynamics, duration, and evidence limits.

### 3. Ground Access From A TLE

```text
Create a short public scenario that initializes a satellite from a TLE and
checks visibility from Colorado Springs. Validate, run, query the review store,
and summarize access windows. Be explicit about whether this is SGP4.
```

Expected agent behavior:

- state that OEL uses the TLE to initialize an ECI state and then numerically
  integrates the configured force model,
- avoid claiming OGP-SGP4/general-perturbations propagation for this initializer-only
  task,
- use the `ground_access` review table when present,
- report access duration, sample count, ranges, elevations, or no-access
  evidence from saved artifacts.

### 4. Attitude Hold

```text
Create a public attitude-hold scenario with one satellite starting with a
pointing error and body rates. Validate, run, inspect artifacts, and explain
what evidence supports the attitude-control result.
```

Expected agent behavior:

- use a public attitude controller,
- keep the orbit problem simple unless attitude/orbit coupling is requested,
- inspect `object_state`, `metrics`, `events`, artifacts, and any attitude
  summary available from the run,
- avoid claiming high-fidelity ADCS validation.

### 5. Compare One Scenario Change

```text
Take the closed-loop rendezvous case and change only the initial in-track
separation. Validate both cases, run both, query the review evidence, and
compare final range, closest approach, burn activity, and limitations.
```

Expected agent behavior:

- change one variable at a time,
- use distinct scenario names and output directories,
- keep validation and run commands visible,
- compare metrics from artifacts or review queries,
- avoid overclaiming if the run duration, controller, or outputs are not
  sufficient for the conclusion.

## Evidence Checklist

A successful agent run should identify the output directory and inspect the
relevant artifacts:

- `index.md`
- `master_run_summary.json`
- `review/run.sqlite` when `outputs.review.enabled: true`
- `review/schema.json` when review output is enabled
- generated CSV files when requested
- generated plots when requested

The agent should cite the exact command or SQL query used to support important
claims.

## Red Flags

The evaluation should fail or be marked partial when the agent:

- skips config validation,
- runs untrusted scenario YAML,
- invents physics or metrics outside OEL artifacts,
- silently enables high-fidelity force models, sensing, estimation, campaigns,
  plots, or animations that the user did not request,
- claims SGP4 propagation from a TLE-initialized OEL run that does not
  explicitly set `propagation_method: general` and `general.model: sgp4`,
- claims terminal rendezvous from partial closure,
- summarizes a completed run without inspecting saved artifacts,
- claims `review/run.sqlite` exists when the run did not create it,
- omits limitations for validation, model fidelity, or missing outputs.

## Feedback Questions

After the evaluation, ask:

- Which prompt produced a useful result?
- Which prompt produced an answer you would not trust?
- Did validation catch or prevent any mistakes?
- Were review-store queries useful compared with JSON or plot inspection?
- Which artifact would you want in an engineering review?
- Which missing evidence would block serious use?
- What first workflow should OEL make easier for agents next?

## Result Classification

Use these labels:

- `pass`: the agent validated, ran, inspected artifacts, and answered from
  evidence.
- `partial`: the agent completed the run but missed a workflow, evidence, or
  limitation requirement.
- `fail`: the agent skipped validation, invented results, used unsafe inputs,
  or made unsupported claims.
