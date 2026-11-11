# Offline And Air-Gapped Installation

An offline release bundle is a ZIP containing the signed release manifest,
source artifact, trusted public release keys, checksums, installers, and the
content-bound supply-chain evidence shipped with that release.
Official offline bundles are platform/architecture/Python-minor specific and
contain an exact signed wheel inventory. The filename ends in a tag such as
`arm64-py311.bundle.zip`; use the bundle whose Python tag matches the OEL
launcher that runs `install-bundle`. Runtime creation checks that tag's signed
qualification before invoking pip, then forces pip `--no-index` against that
wheelhouse. A wrong Python minor or a missing, extra, changed, or incomplete
wheelhouse fails closed instead of contacting a package index.

On a connected verification host, inspect the bundle inventory, verify the
published bundle digest through the approved out-of-band channel, and transfer
it using local policy. On the target host:

Before the first direct install on a target, verify the bundled
`trusted-release-keys.json` through the same approved channel and place it at
`trusted-release-keys.json` beneath the selected OEL config root. The config
root is the platform default unless `--config-root` is supplied. The
`--public-keys` argument below may point to that file, but an explicit key-file
argument does not replace or initialize the external trust anchor.

```text
oel update install-bundle <oel-public-version.bundle.zip> --public-keys <trusted-release-keys.json>
oel update activate <version>
oel update status --full
oel doctor
```

The bundle path performs no network operation. Archive traversal, links,
devices, duplicate members, excessive expansion, signature failure, hash/size
mismatch, wrong platform, unsupported Python, and source/manifest version
mismatch are rejected before activation.

The v1 local edition does not require a product license or activation. External
runtimes such as cFS, MATLAB, GPU drivers, or ML stacks are reported as
qualified prerequisites; their absence does not broaden core claims or cause
the updater to fabricate replacements.

Rollback and cleanup are local operations. Retain at least the current and
previous engine, every workspace-pinned engine, the signed bundle, and its
published digest according to site policy.
