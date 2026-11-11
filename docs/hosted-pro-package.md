# Hosted Execution Package Contract

**Hosted OEL is a closed alpha; access is not publicly available.** Public OEL
remains free to run locally without a Hosted account. The public package
validator is an offline local check and does not grant service access. Any
private alpha access requires separate operator provisioning and current
service instructions; this package does not establish service availability.
The acceptance proof uses simulated settlement, not real payments.

Public OEL can prepare and locally validate an analysis package that may be
considered for local or Hosted-only execution through Hosted OEL. Local
packages remain runnable locally for free. A locally valid Hosted route does
not establish an available venue. Hosted-only packages require a qualified
Hosted executor because the public installation does not contain or import
service implementations. Both reuse the versioned OEL StudyRequest and
StudyPlan contracts rather than introducing a second planning language or
accepting arbitrary shell commands.

Local package validation is free, read-only, non-executing, and offline. A
valid local receipt means the selected bytes, public contract, typed operation
graph, declared resource bounds, and transfer inventory agree. It does not
establish remote availability, entitlement, price, scientific qualification,
or execution approval. When an authorized Hosted service is operating, it must
repeat validation over the exact sealed bytes with its authoritative validators
before returning a quote.

## Package Layout

```text
my-study/
  oel-hosted-package.json
  request.json
  plan.json
  inputs/
    problem.json
    observations.csv
```

`oel-hosted-package.json` uses `oel.hosted_execution_package.v1` and identifies the
request, proposed plan, selected input files, media types, optional scenario
configuration references, and requested retention policy. Its public JSON
Schema is
[`docs/contracts/schemas/oel-hosted-execution-package-v1.schema.json`](contracts/schemas/oel-hosted-execution-package-v1.schema.json).
The validator continues to accept the legacy `oel.hosted_pro_package.v1`
manifest for Hosted-only packages. The `pro` spelling is a compatibility
identifier, not a separate local product or license tier.

The request uses `oel.hosted_study_request.v1`. Required inputs may initially
set `content_sha256` to `null`; the local validator streams each selected file,
binds its actual SHA-256 and byte count, and rejects any conflicting declared
identity. The plan uses `oel.hosted_study_plan.v2`. An agent proposes operation
dependencies, parameters, bounds, evidence requirements, acceptance criteria,
claims, non-claims, and transfers. OEL compiles the named input and output ports
itself rather than trusting agent-authored edge identities.

No package field is a shell command. Operations must use capability IDs and
parameter schemas published by `oel study capabilities --scope discovery`.

## Validate Locally

```bash
oel hosted package-schema
oel hosted validate-package ./my-study --workspace-root .
```

A successful public-only receipt reports both execution venues and a local
recommendation:

```text
status:                     CLIENT_VALID
planning route:             LOCAL_FREE_AVAILABLE
local execution available: true
hosted execution available: false (closed alpha; profile access not established)
recommendation:             local (not required)
hosted preflight required:  true
execution authorized:       false
payment authorized:         false
files uploaded:             false
```

A Hosted-only package instead reports `HOSTED_PRO_REQUIRED`, local execution
unavailable, and no executable venue recommended. Neither route has a price;
these local results do not establish that a live quote can be issued. If an
authorized Hosted service is operating, it must revalidate the sealed package
and issue an immutable quote before execution could be considered.

The receipt contains the bound request, bound plan, typed-edge compilation
receipt, local planning result, immutable input inventory, package digest, and
explicit non-claims. It is suitable for review and for a later content-bound
upload request. It is not a hosted validation receipt or quote.

## Local Checks

The validator:

- rejects absolute paths, traversal, duplicate logical paths, and symlinks;
- rejects duplicate or non-finite JSON fields in control documents;
- streams file hashing instead of loading large inputs into memory;
- binds exact input SHA-256 identities and byte counts;
- checks public schema identifiers and required typed-port metadata;
- validates capability parameters and visible resource bounds;
- safely validates scenario YAML without importing configured plugins;
- requires every operation to contribute to promised evidence;
- accepts a fully public `LOCAL_FREE_AVAILABLE` plan through the generic
  manifest; and
- keeps the legacy Hosted-only manifest fail-closed by requiring at least one
  published Hosted capability and a `HOSTED_PRO_REQUIRED` route.

Control documents are limited to 2 MB. Individual selected inputs are bounded
to 16 GiB in this v1 client contract. A future upload grant may impose a lower
tenant, package, file, or quote-specific limit.

## Validation Levels

| Level | Performed by | Meaning |
| --- | --- | --- |
| Package integrity | Public OEL | Exact local bytes and manifest agree |
| Public planning contract | Public OEL | Request, plan, ports, bounds, evidence, and transfer intent agree |
| Public scenario validation | Public OEL | Scenario shape and plugin-pointer shape pass without imports |
| Authoritative Hosted validation | Hosted OEL | Server-side schema, semantic, entitlement, and worker validation over the exact sealed package |
| Execution approval | User plus Hosted OEL | Exact quote and limits were explicitly approved |

Only the final two levels can make a package executable through an operating,
authorized Hosted service. They do not establish current Hosted availability.
They do not prevent a public-only package from being executed locally.
