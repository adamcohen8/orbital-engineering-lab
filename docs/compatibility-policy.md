# OEL 1.x Compatibility Policy

This is the compatibility policy for the open-source local OEL edition,
starting with 1.0.0. It governs the supported surfaces below. Component
contracts define their exact fields, units, limits, and maturity. Platform
qualification and dependency profiles are covered separately by
[Compatibility](compatibility.md).

## Supported surfaces

| Surface | Supported contract |
| --- | --- |
| Python | The documented imports, call signatures, return semantics, and compatibility aliases in the [Python API](python-api.md), [configuration/API architecture](config-api-architecture.md), and linked public workflow guides. This includes the scenario, session, workspace, result, snapshot, and validation façades exported by `sim` and `sim.api`; the documented `sim.config`, `sim.execution`, and `sim.review` interfaces; and the standalone APIs linked from [local workflows](v1-local-workflows.md). |
| CLI | Documented commands, existing valid arguments, and exit-status semantics for `oel`, `oel-sim`, and the local workflow module CLIs. Consume versioned JSON or artifact manifests for automation; console prose, progress displays, and help layout are not serialization contracts. |
| Scenario configuration | The [scenario YAML contract](contracts/scenario-yaml-contract.md), including accepted `oel.scenario.v1` fields, units, defaults, validation semantics, and normalization. The currently supported unversioned/`oel.scenario.legacy.v0` normalization remains available throughout 1.x; this does not reinstate rejected historical shapes. |
| Results and evidence | The documented stable fields and units in the [payload/artifact contract](contracts/payload-artifact-contract.md), versioned workflow products, and [review-store contract](review-store.md). Discover optional artifacts through manifests and query named SQL columns. |
| Extensions | Documented plugin-pointer formats and extension protocols, including the [flight-software authoring and FSWDK workflows](fsw-authoring.md), within their declared maturity and trust boundaries. A plugin remains trusted executable code. |
| MCP | Published tool IDs and their versioned request, result, effect, and evidence contracts in [OEL MCP](oel-mcp.md). Tool discovery reports availability, maturity, and deprecation independently. |

The documented stable Python façade classes include `SimulationConfig`,
`ScenarioArtifact`, `ScenarioBuilder`, `SimulationSession`,
`TrustedSimulationSession`, `HostedSimulationSession`,
`SimulationWorkspace`, `TrustedSimulationWorkspace`,
`HostedSimulationWorkspace`, `SimulationResult`, `MetricStudyResult`,
`SimulationSnapshot`, `ValidationIssue`, and `ValidationReport`.
Documented names containing `pro` remain supported local import/CLI names;
they do not require local license activation.

An importable module or a name in `__all__` alone does not promote an
implementation detail to a stable API. Undocumented helpers, underscore-prefixed
internals, `sim.public_api` implementation owners, `_SingleRunEngine`, debug
records, and internal native binding layouts are not public extension contracts.
Compatibility façades continue to preserve documented imports and class identity.
Explicitly experimental or prototype surfaces retain their stated maturity;
their changes require release notes and migration guidance but may occur in a
minor release. Public Hosted client contracts do not promise service availability.

## Release and schema versions

- Patch releases preserve supported interfaces while correcting defects.
- Minor releases may add optional parameters, commands, fields, tables, and
  capabilities. Existing valid callers retain their documented behavior.
  New required arguments, removed fields, changed units/meanings, or changed
  defaults that alter valid existing studies are breaking changes.
- Breaking changes to stable surfaces require a new OEL major release.
  A separately versioned tool or product may introduce a breaking successor in
  1.x only while the supported predecessor remains available for 1.x consumers.
- Schema and tool versions evolve independently of the OEL package version.
  For example, review schema `0.13` does not become `1.0` merely because OEL
  becomes 1.0. A schema bump does not by itself authorize breaking a supported
  1.x reader or silently reinterpreting an existing product.
- Readers must inspect schema IDs and discover optional fields/tables.
  Old readers need not accept future schemas or newly introduced features.
  Unknown scenario versions continue to fail closed.

New review stores declare `stable_1_x_additive`. Stable tables listed in their
`schema.json` preserve existing column names, units, and meanings throughout
1.x. Additive tables and nullable columns are allowed. Stores previously marked
`pre_1_0_additive` keep their original metadata; publication never rewrites
completed evidence. Their documented read compatibility remains supported.

## Deprecation and corrections

A stable surface deprecated during 1.x remains available through the 1.x line.
Record the first deprecated release, replacement, migration example, and
earliest removal major release in the owning guide and changelog. Emit a
deprecation warning or versioned discovery metadata where supported. Removal
requires a major release and documented migration.

Security fixes may reject unsafe inputs or close an unintended trust bypass
within 1.x. Validation fixes may reject inputs already outside the documented
contract. Document affected cases and migration in the release notes; do not
treat either exception as permission to redesign supported interfaces.

Numerical correctness fixes may change results within a patch or minor release.
Record affected models, source/config/dependency identities, validation
tolerances, and observed deltas; run the owning regression and reference checks.
Preserve units and meanings. Bitwise identity across changed implementations,
dependencies, or platforms is not promised. Deterministic replay applies within
the recorded execution envelope, and numerical parity does not establish
empirical or operational qualification.

## Change acceptance

For a supported-surface change, update the owning contract and changelog,
retain regression coverage for existing callers/products, and test the generated
public distribution. Configuration inspection and plugin-policy tests must also
work in a fresh process, without imports from an earlier test supplying state.
Use the release candidate gate before freezing a changed release; historical
passing evidence is not qualification for a different source tree.
