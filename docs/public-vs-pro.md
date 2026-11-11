# Public OEL and private services

Starting with v1.0, OEL has one open-source local edition. The package retains
its dependency profiles and existing module names, including legacy names
containing `pro`. Local workflows require no license activation.

See [OEL v1.0 local workflows](v1-local-workflows.md) for campaign and parallel
execution, covariance and conjunction, trajectory and constellation design, RF,
tracking OD and SLR, controller benchmarking, and spacecraft flight software.
[Installation](installation.md) describes the dependency profiles.
The [validation guide](validation-claims.md) explains how to interpret evidence
and where the release provides recipes and redistributable inputs.

The public client and data contracts for Hosted OEL do not include or grant
access to the hosted service. Service infrastructure, subscription workflows,
and the Manifold remain private. Provider-driven config generation and reports,
IHE, Frontier/Mendicant evaluation, Community, prototype Trainer levels, and
generated validation evidence remain private. Use [the local workflow MCP](oel-mcp.md)
to prepare evidence and audit an agent-authored report. The local simulator
remains usable offline.

Public releases are generated from a reviewed allowlist. The development
repository, private `agents/pro/` instructions, private inputs and generated
validation evidence are outside the public release. Use public-safe inputs
when sharing an example or reporting an issue.

For platform and evidence limits, see [Known Limitations](known-limitations.md).
Planetary numerical propagation follows the [bounded planetary contract](planetary-onp.md),
with user-supplied environmental data where the release does not provide it.
