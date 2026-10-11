# OEL Rust numerical core

This crate contains the numerical implementations behind OEL's public Python
bindings: ONP and OGP propagation, CR3BP and relative dynamics, attitude,
estimation, controls, coverage and free-space links, spacecraft and vehicle
models, and supported standalone analysis primitives.

See [building public native source](../../docs/native-source-build.md) for
Cargo tests, wheel builds, installation, dependencies and qualification limits.
Python retains configuration, workflow and evidence authority. A native
kernel's presence does not establish a supported product workflow.

The generated public manifest builds only the public source owners. Private
source builds retain a separate `pro` feature for their additional native
owners; these owners are absent from the public repository. Both builds use
one authoritative source implementation for each shared kernel.

Orbit states use km and km/s; times use seconds. Other numeric boundaries retain
the units documented by their Python owners. The SGP4 dependency and the
NRLMSISE-00 implementation retain their upstream licenses and provenance in
`vendor/` and `src/vendor/`. Additional force-model attribution is in
[NOTICE.txt](NOTICE.txt); repository code is Apache-2.0.
