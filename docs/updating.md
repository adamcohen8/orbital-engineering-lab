# Updating OEL

OEL installs engines side by side. Downloading or globally activating a newer
engine never rewrites scenario YAML, flight-software source, dependency locks,
tests, or prior outputs. A registered workspace keeps its exact engine pin
until its owner audits and explicitly adopts another version.

The product display name is Orbital Engineering Lab and new Python builds use
the `orbital-engineering-lab` distribution name. Existing `oel` commands and
managed data directories keep their names. Signed v1 release manifests retain
the `orbital-engagement-lab` product field so previously installed updaters can
accept a new release; new readers also accept the branded value. Existing RPO
Trainer progress and settings are read in place from `~/.orbital_engagement_lab`
when no corresponding file exists under `~/.orbital_engineering_lab`. New files
use the new directory. Public schema files use the engineering-name Vercel URLs;
previously produced packets can still carry their original schema IDs. The
desktop app bundle ID stays stable for update compatibility.

The public project website is `https://orbital-engineering-lab.vercel.app/`.
The former `https://orbital-engagement-lab.vercel.app/` address permanently
redirects to it. Older GitHub repository links continue through GitHub's
repository rename redirect; new installation instructions use the renamed
repository directly.

## Check And Install The Latest Release

These commands apply to the v1.0.0 or newer managed launcher. A v0.36.0
installation already has the signed-inventory trust boundary and can update
directly. For a v0.31.0 through v0.35.0 installation, first rerun the official
installer as described in [installation](installation.md) to establish that
boundary.

```text
oel update status --full
oel update check
oel update install latest
```

The official installer persists the stable signed-channel URL and trusted
release keys. The release-key file is an operator/channel trust anchor beneath
the platform config root, outside every version directory. `check` therefore
needs no URL or key arguments for an ordinary managed installation. It performs
a network read and rejects an invalid signature, wrong edition/channel,
expired or revoked trust key, or release-feed rollback. `install latest` checks
that channel, downloads the exact referenced manifest and artifact, verifies
their identity, and installs the release side by side. It does not activate the
release or edit a workspace.

Official manifests also carry a signed source and runtime inventory. Activation
and managed dispatch verify the extracted source bytes, wheel metadata and
installed `RECORD` files against that inventory. The installation record and
`installed-tree.json` remain useful local diagnostics but cannot establish
official trust. A release created before this inventory was introduced is
retained for inspection and reported as `untrusted`/`reverification_required`;
it is never made official by hashing its current tree locally.

The complete inventories use a lossless XZ/base64 transport inside the signed
manifest, with a 32 MiB decoded bound and a 64 MiB decoder memory bound. The
standalone bootstrap and managed verifier use the same decoder implementation.
The existing 4 MiB manifest download limit remains in force, and the builder
rejects a manifest that exceeds it. Decoding preserves all source, wheel,
profile and runtime digests; it does not modify the signed manifest.

An installation created before channel persistence can configure its official
endpoint once:

```text
oel update configure-channel https://github.com/adamcohen8/orbital-engineering-lab/releases/latest/download/public-stable.json
oel update check
```

Operators may still use the lower-level explicit forms for mirrors and release
diagnostics:

```text
oel update check --channel-url <signed-channel-url> --public-keys <trusted-keys.json>
oel update download <signed-release-manifest-url> --public-keys <trusted-keys.json>
oel update install <cached-release-manifest.json> --public-keys <trusted-keys.json>
```

`download` retains a bounded partial file for retry, then verifies exact size
and SHA-256. None of these commands activates a release or edits a workspace.
Key rotation is explicit: a new `oel.trusted-key-registry.v1` must be signed by
a currently trusted, non-revoked key and is installed with
`oel update rotate-trusted-keys <registry.json>`.
An incoming bundle may carry a future embedded key list, but its manifest must
still verify against the current anchor until that signed rotation is
completed. Revoking a release signer in the current anchor also prevents
activation and dispatch of installations signed by that key. The installer never
replaces an existing anchor merely because a bundle contains a different registry.
For an installation created before replay timestamps existed, the one
supported migration baseline is the external bare `{"keys": [...]}` anchor;
the first valid signed rotation must include a timezone-aware `published_at`
and establishes the replay timestamp. A versioned or malformed anchor without
that timestamp is rejected and requires explicit operator repair.
Existing installations remain eligible after an authorized non-revoking
rotation: the current external anchor re-verifies the signed manifest and
inventory, while the installation record's `trust_anchor_sha256` is retained
as historical provenance and may differ from the current anchor after
rotation. A changed historical digest is reported as a diagnostic; signer,
manifest, integrity, runtime, and external receipt mismatches still fail
closed.

## Install And Activate

```text
oel update activate <version>
oel update rollback
```

Install extracts into an incomplete transaction directory, validates host and
Python compatibility, checks the archive source version against the signed
manifest, creates a constrained environment, runs `pip check`, records
provenance against the external trust anchor, and only then publishes the
version.
Activation replaces the stable launcher and current selector as a recoverable
transaction. The launcher invokes an operator-owned Python in isolated mode,
which verifies the external key anchor, signed source inventory, and managed
runtime inventory before importing the selected version. Desktop entries use
the same launcher gate. Global activation does not change registered workspace
pins. Official provisioning uses `pip --no-compile` and removes only
provisioning-time derived caches from its trusted staging runtime before the
signed check; any bytecode cache found in the published source or runtime is
an authentication failure.

The v1 local edition does not require a product license or activation. Hosted
execution, when available, is separately provisioned by the service and is not
part of the local updater or workspace migration contract.

## Audit And Adopt Per Workspace

```text
oel workspace check path/to/workspace --against <version> --release-manifest <manifest.json>
oel workspace migrate path/to/workspace --to <version> --release-manifest <manifest.json>
oel workspace migrate --apply-plan <migration-plan.json>
oel workspace use path/to/workspace <version>
oel workspace rollback path/to/workspace
```

The audit reads YAML, manifests, lock metadata, and prior evidence without
importing candidate code, executing a simulation, or using the network.
Possible dispositions are `compatible`, `compatible_with_warnings`,
`migration_available`, `manual_review`, `blocked`, `invalid`, `incomplete`,
and `cancelled`. `workspace use` accepts only installed verified engines and
an audit status of `compatible` or `compatible_with_warnings`. Migrations require a content-bound plan and explicit apply;
backups and a receipt remain beneath `.oel/migrations/`.

## Cleanup And Uninstall

```text
oel update cleanup
oel update cleanup --apply --keep 2
oel update uninstall <version>
oel update uninstall <version> --apply
```

Both commands default to a dry run. Current, previous, workspace-pinned, and
process-leased engines are protected. Uninstall never deletes registered
workspaces. Removed engines are recoverable by reinstalling the same verified
release artifact; user work is not part of an engine installation.

For a sanitized diagnostic artifact, run:

```text
oel support-receipt oel-support-receipt.json
```

The receipt omits workspace paths, user source, config contents, and customer
data and sends no telemetry.
