# Agent Run Lifecycle: Tip And Nap

OEL exposes a transport-neutral lifecycle for one trusted, deterministic,
foreground scenario. An agent or harness can start the run, retain an opaque
handle, stop spending model turns supervising it, and perform one bounded
await. The wake result points to committed durable state; it is not simulation
evidence by itself.

The contract is available through the `oel runs` CLI and the
`sim.execution.run_lifecycle` Python package. It has no dependency on MCP or a
particular model provider, so Codex, Claude Code, Grok Build, and local
harnesses can use the same JSON records.

## Safety Boundary

Lifecycle v1:

- executes OEL scenario YAML only, never an arbitrary shell command;
- requires explicit local invocation and trusted plugin validation;
- accepts one deterministic single-scenario foreground run;
- forces the `laptop-safe` resource profile and serial public execution;
- disables live AI report and AI-config providers;
- requires a new or empty output below an authorized output root;
- enables the standard review store;
- caps each await at 3,600 seconds; and
- never changes run state from `inspect`, `events`, or `await`;
- records a content-digested local execution-owner heartbeat; and
- requires an explicit `reconcile` operation before owner loss becomes
  `interrupted`.

There is no background daemon, queue, automatic retry, cancellation endpoint,
release-workflow integration, or MCP adapter in v1. The process invoking
`start` remains the execution owner.

## CLI Workflow

JSONL mode flushes the accepted handle before execution begins, then writes a
terminal result when execution ends:

```bash
oel runs start \
  --config configs/automation_smoke.yaml \
  --output-dir my-agent-run \
  --jsonl
```

Preserve the first record's `run_id`, `manifest_ref`, and
`normalized_config_sha256`. From another process, perform a bounded await:

```bash
oel runs await RUN_ID \
  --timeout 900 \
  --expect-manifest-ref MANIFEST_REF \
  --expect-config-sha256 CONFIG_SHA256
```

`terminal` means OEL verified a terminal manifest already committed in the run
output. `still_running` is a normal timeout, not a failed run. `not_found`,
`identity_mismatch`, `malformed_state`, and `observer_error` remain distinct.
If the foreground process exits before terminal commit, `await` returns
`owner_lost` while leaving state unchanged. Reconcile that verified local
owner loss explicitly:

```bash
oel runs reconcile RUN_ID \
  --expect-manifest-ref MANIFEST_REF \
  --expect-config-sha256 CONFIG_SHA256
```

Successful reconciliation commits terminal state `interrupted` and event
`interrupted`. Repeating it is safe and returns `already_terminal`. A live,
missing, or non-local owner is never converted to `interrupted`.

Read current state or ordered events without waiting:

```bash
oel runs inspect RUN_ID
oel runs events RUN_ID --after-sequence 2
```

Events are monotonically sequenced and at-least-once. Deduplicate with
`(run_id, sequence)`. The manifest is authoritative; the locator and event
stream are discovery and notification surfaces.

## Durable Files

Each output contains:

```text
<output>/lifecycle/run_manifest.json
<output>/lifecycle/run_events.jsonl
<output>/lifecycle/execution_owner.json
<output>/lifecycle/execution.log     # JSONL CLI mode
```

The manifest has a canonical self-digest. Terminal states are immutable, and
OEL writes the manifest before emitting the corresponding event or updating
the non-authoritative locator. A wake notification therefore tells the caller
to inspect committed state; it does not replace review queries or artifact
validation.

The local lifecycle store retains the authorized state and output directories
through descriptor-bound, no-follow handles. Locators, manifests, events,
owner records, and lock publication/reclamation are opened relative to those
handles; a descendant symlink, unsupported secure-path primitive, or visible
replacement of a retained root fails closed. The macOS `/tmp` and `/var`
compatibility aliases are canonicalized only at the platform root before the
component walk. The lifecycle store keeps the existing JSON schemas, event
ordering, and stale-owner recovery rules while refusing to use a replacement
directory that happens to reuse the same visible pathname.

## Provider-Neutral Reference Client

`examples/python/run_lifecycle_client.py` demonstrates the subprocess pattern
without an agent SDK:

```bash
python examples/python/run_lifecycle_client.py \
  --config configs/automation_smoke.yaml \
  --output-root outputs \
  --output-dir lifecycle-example \
  --state-root .oel/run-lifecycle \
  --timeout 900
```

For long waits, let the host wait for command completion instead of repeatedly
waking the model to poll status. A normal bounded timeout does not indicate failure.

A host-native adapter can translate its own sleep or wake primitive around the
same `start`, `await`, and `inspect` records. MCP may expose these operations in
a later layer, but it is not the lifecycle's foundational protocol.
