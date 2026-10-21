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

## Planetary ONP (wheel 0.18.2)

`src/planetary.rs` is a separate Rust force plan for body-centered J2000 states.
It prepares gravity and atmosphere rotations, normalized static harmonics
through degree 20, bounded exponential/log-density tables, cubic Hermite
ephemerides, drag, SRP and differential external gravity. RK4/RKF78/DOPRI5
stages stay in Rust; there is no Python mirror or stage callback. Earth
`ForcePlan` codes/layouts/bindings are unchanged. Python owns configuration,
local source validation, lifecycle, serial stepping and neutral artifacts.
The native `PlanetaryGravityDomainError` distinguishes a primary harmonic
reference-sphere exit from other model failures. The planetary lifecycle
discards that serial interval for all objects and reports the last valid
sample; it does not infer physical impact or evaluate an interior field.

Build with the existing maturin workflow (`maturin develop --release` from
this crate in an activated environment), or build a wheel and install it in
the intended runtime. `cargo test --manifest-path rust/oel-orbit/Cargo.toml`
from the repository root also exercises the standalone numerical core. Do
not replace another managed runtime's installed wheel during development.
See [Planetary ONP](../../docs/planetary-onp.md) for model/data contracts and
the optional SciPy/CSPICE reference qualification command.

## Extended external fields (wheel 0.18.2)

`src/additional_gravity.rs` composes static external spherical harmonic fields
through degree 20. Each source owns its GM, radius, normalized coefficients and
rotation; a geometric Hermite ephemeris supplies its physical center relative
to the simulation origin. The evaluator subtracts the full field at the origin
from the full field at the spacecraft, including both monopole and harmonics.
It rejects interior/singular geometry and out-of-domain stages.

`AdditionalGravityPlan` exposes optional Earth and planetary composition
bindings with RK4/RKF78/DOPRI5 integration. Earth composition consumes the
existing `ONPForceContext` and native stage preparation. An external Earth
field can also consume the existing native EOP frame context. No additional
source branch enters the ordinary Earth force evaluator or integration loops.
See [extended third-body gravity](../../docs/additional-gravity.md).

## Bounded OGP batch workers

The native batch adapter can allocate bounded object-row workers and reports
requested, allocated, and used workers. Queries within each object remain
ordered; small batches and single-object batches stay serial. Scale worker
selection remains a Pro workflow. This does not change public single-scenario
serial stepping or establish a universal throughput improvement.
