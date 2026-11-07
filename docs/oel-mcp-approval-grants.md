# OEL MCP approval grants and durable effects

OEL MCP effect tools accept host-configured operator approvals. An approval is
an opaque selector for authority that the host already provisioned; request
content cannot create, widen, or renew a grant. The model cannot mint a grant
by placing a new identifier in an MCP request.

## Standing approvals and bounded grants

The existing environment variables remain valid for compatibility:

- `OEL_MCP_WRITE_APPROVAL_IDS`
- `OEL_MCP_EXECUTION_APPROVAL_IDS`
- `OEL_MCP_TRUST_APPROVAL_IDS`

A standing approval with only `approval_id` and `scope` preserves the existing
local behavior. It is not retry-safe by itself. A request may include the
optional `operation_id` field when the host has configured a durable ledger;
that turns the effect into an idempotent operation with exact request identity.

Bounded grants are configured by the host through
`OEL_MCP_APPROVAL_GRANTS_JSON` (the `OEL_MCP_APPROVAL_GRANTS` name is accepted
as a compatibility alias). A bounded grant must include an exact tool ID, an
expiry, and either the SHA-256 digest of the reviewed arguments or a reviewed
plan operation together with host-verified content identities. A grant may
also carry resource ceilings. Request fields are never resource
evidence. After exact request and content binding succeeds, the supported OEL
producer must pass a host-computed resource plan or estimate before its first
effect. A missing plan or metric, or an unsupported producer path, fails closed;
a caller cannot satisfy a ceiling by supplying a model-controlled value.

A minimal exact-argument grant looks like this:

```json
{
  "grants": [
    {
      "grant_id": "review-2026-01",
      "scopes": ["write"],
      "tool_ids": ["oel.materialize_scenario_patch.v1"],
      "reviewed_arguments_sha256": "<64 lowercase hex characters>",
      "expires_at": "2026-12-31T23:59:59Z",
      "purpose": "one reviewed local materialization"
    }
  ]
}
```

The argument digest is canonical JSON over the tool ID and all request
arguments except `approval` and `trust_approval`. A changed path, option,
resource request, or payload therefore fails before the operation starts.

For a plan-bound grant, `operation_id` must match exactly when supplied by the
request. The host may bind the grant to a file with a `content_identities`
entry such as `{"argument": "config_path", "kind": "file", "sha256":
"..."}`. OEL reads that file through the authorized read-root policy with
no-follow checks and computes the digest itself. `review_digest` identities
are accepted only when the host supplies a trusted resolver; a caller-supplied
review hash is never trusted as evidence. Revoked grant IDs are supplied with
`OEL_MCP_REVOKED_APPROVAL_GRANTS`, and in-process revocation also fails closed.

Approval scope is checked before grant matching. A write grant cannot be used
for execute or trust, even when its identifier is otherwise valid.

## Durable operation ledger

Bounded effect grants always require a host-private durable ledger. A standing
approval becomes durable only when the request supplies `approval.operation_id`.
Set `OEL_MCP_OPERATION_LEDGER_ROOT` to an absolute directory owned by the host
user. The directory must be outside every MCP read and write root, and OEL
creates it with mode `0700`. Injected ledgers follow the same separation rule.
OEL deliberately does not create a ledger inside a model-authorized workspace.

The ledger reserves an operation before the effect callback runs. It records
the tool, authority, and canonical request digest. A retry with the same
identity returns the saved terminal envelope without running the callback a
second time. A changed request is an operation conflict. If the process stops
after reservation and before terminal commit, a retry returns
`UnknownOperationOutcome`; the effect is not rerun automatically.

Completed records retain hashes for file artifacts exposed by the result. A
retry verifies those files before replay. If an artifact is missing, changed,
or could not be hashed, OEL returns the stored result with `partial` status,
`evidence.complete=false`, and a `StoredArtifactStale` error so the result is
not presented as current evidence.

### Ledger filesystem boundary

OEL opens the explicit host ledger root once during process initialization and
retains its directory descriptor for the lifetime of the MCP process. Per
operation locks, bounded record reads, and new or atomic record publications
are relative to that descriptor. Renaming the visible ledger root after
initialization cannot redirect a live operation's writes or erase an unresolved
`begun` record. Failed or unknown-outcome temporary records are retained when
the filesystem cannot establish safe ownership of the name.

This boundary preserves the host's initial trust decision for the configured
root; it does not authenticate the root's identity or ownership across a
process restart. The ledger lock is advisory and serializes cooperating
writers only. A hostile same-account process that ignores the lock can still
replace a directory entry, and protection from arbitrary code execution or
same-UID `ptrace` is outside the ledger contract. Callers must continue to
inspect `UnknownOperationOutcome` and reconcile effects explicitly.

Each retained operation has a zero-data `.<operation-id>.lock` companion that
is kept for the same lifetime as its durable operation record. OEL does not
auto-expire or unlink that guard while the ledger is active: retiring lock
names early would allow an old writer to split from the record's advisory
serialization. The finite set of one lock companion per retained record is
ledger metadata, not an independent payload cache. The operator may retire
the whole ledger only after closing all owners and accepting retirement of
the idempotency history with it.

Secure writers can read the reserved ID through the handler's
`active_operation_id` property and associate it with their host-owned output
manifest. The ledger and writer transaction remain host responsibilities;
MCP transport and models do not grant spacecraft, provider, or external
operational authority.
