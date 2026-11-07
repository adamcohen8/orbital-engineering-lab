# Installer And Updater Threat Model

## Protected Assets

The managed installer protects release identity, immutable engine files,
launcher selection, workspace ownership, prior evidence, trusted release keys,
and Pro entitlements. Scenario YAML and FSW source are untrusted user inputs
until explicitly trusted for execution; they are never updater-owned.

## Trust Boundaries

- The initial shell or PowerShell script is transport-fetched and should be
  inspected. It verifies the digest of the larger standalone bootstrap.
- The bootstrap embeds an offline public release-key registry. Release signing
  keys are separate from license keys and are never shipped privately.
- Signed channel metadata names a signed release manifest; the manifest binds
  edition, version, contracts, artifacts, sizes, hashes, and compatibility.
- HTTPS update hops pin DNS results and retain certificate/hostname checks.
  Redirects remain refused except for at most three hops along this public
  repository's canonical GitHub release route, including its repository-ID-bound
  `release-assets.githubusercontent.com` path. Each destination is independently
  resolved and pinned; authorization and cookie headers are not forwarded.
- The signed manifest also binds a deterministic source inventory and the
  wheel archive members, metadata, installed-`RECORD` expectations, generated
  script mappings, and managed interpreter layout used by the runtime proof.
- The current release-key registry is an external operator/channel anchor
  beneath the persistent config root, outside version directories. A bundle
  with a different embedded registry cannot replace it; adoption requires an
  explicitly signed rotation by the current non-revoked authority.
- The rendered official installer persists its HTTPS channel endpoint as local
  configuration. The endpoint is only a locator: every channel response and
  referenced release remains signature-verified against the separate trusted
  key registry, and channel-to-manifest edition/version identity must agree.
- Extracted source is not imported until signature, artifact, and archive-shape
  checks pass. Ordinary workspace audits do not import user plugins or FSW.
- Official source and managed-runtime trees reject ``__pycache__``, ``.pyc``,
  and ``.pyo`` files. Provisioning uses pip ``--no-compile`` and removes
  provisioning-time derived caches from its trusted staging runtime; disabling
  writes alone is insufficient because Python can still import a forged
  pre-existing cache.
- Activation writes the launcher with an operator-owned Python and a small
  stdlib-only verifier bundle beneath the external launcher directory. That
  verifier authenticates the external key anchor, signed manifest, source
  inventory, and runtime inventory before the managed ``sim`` package is
  imported. Desktop launchers route through this same gate.
- Activation and workspace adoption are separate transactions.

## Addressed Threats

The implementation rejects metadata/artifact tampering, unknown/expired/revoked
keys, feed rollback, wrong edition/channel/host/Python, source-version mismatch,
oversized downloads, archive traversal, drive paths, links, devices, duplicate
members, expansion bombs, concurrent state mutation, modified official source,
cleanup of selected/pinned/leased engines, workspace path escape, stale
migration plans, and ineligible Pro versions.

State files and selectors use same-directory atomic replacement. Installs use
side-by-side incomplete directories and become selectable only after checks.
Legacy signed manifests without the authenticated inventory remain retained but
are marked untrusted and require explicit re-verification/migration.
Migration plans bind original and proposed digests, create backups first, and
restore already-applied files on failure. Cleanup and uninstall are dry-run by
default and preserve workspaces.

## Residual Risks And Operational Controls

- A compromised trusted signing key can authorize malicious release code.
  Keep private keys offline/HSM-backed, publish key rotation out of band, and
  mark compromised public keys revoked.
- A malicious but correctly signed dependency or release remains trusted by
  the cryptographic layer. Exact-profile supply-chain and validation gates
  remain release-blocking.
- Local configuration can redirect an update check to another HTTPS endpoint.
  Redirected metadata still needs a trusted signature. Use
  `oel update configure-channel` only with an operator-approved endpoint and do
  not place bearer tokens or other secrets in channel URLs.
- The one-line `curl | sh` form hides inspection. OEL documentation uses
  download, inspect, then execute; the shorter pipeline is a convenience only.
- Local administrators and malware with write access to managed roots can
  replace launchers or state. Activation and managed-engine dispatch verify
  source and runtime bytes against the release-signed inventory and external
  anchor; the interpreter provider and the stable verifier bundle outside the
  managed virtual environment are explicitly operator-owned and require
  separate host qualification. This is not an OS integrity monitor.
- Disk exhaustion and power loss can leave incomplete cache/transaction data.
  They must not change the current selector; stale-lock recovery requires
  confirming no updater is active.
- External runtimes, proxies, mirrors, and managed Python provisioning need
  separate qualification. They are not silently enabled by the core updater.

No telemetry is sent. Update checks and downloads occur only through explicit
commands. Support receipts are local and omit workspace paths and contents.
