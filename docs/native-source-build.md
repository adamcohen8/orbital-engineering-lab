# Building public native source

The public repository includes the Rust implementations used by public OEL
numeric routes and the RPO Trainer. Python owns configuration, workflows,
resource loading, event and mission policy, provenance, and evidence artifacts.
The public-source export includes both native crates, their Cargo lockfiles,
Python build manifests, and the required vendored dependency source and licenses.
Pro-only native owners are omitted from the public crate's build.

## Build and install

Use an activated Python environment supported by `docs/installation.md`.
Install the Rust toolchain and Maturin, then run from the repository root:

```bash
python -m pip install "maturin>=1.10,<2"
cargo test --locked --manifest-path rust/oel-orbit/Cargo.toml
cargo test --locked --manifest-path rust/oel-game/Cargo.toml
python -m maturin build --release --locked --manifest-path rust/oel-orbit/Cargo.toml --interpreter python --out dist/native
python -m maturin build --release --locked --manifest-path rust/oel-game/Cargo.toml --interpreter python --out dist/native
python -m pip install --no-index --find-links dist/native "oel-rust-orbit>=0.18.2,<0.19" "oel-rust-game>=0.5,<0.6"
python -m pip install .
python run_simulation.py --doctor
python run_simulation.py --quickstart --validate-only
```

Cargo and pip may download declared public dependencies. To build offline,
prepare those dependencies first and add `--offline` to Cargo/Maturin commands.
The wheels are specific to the build host; a successful local build does not
establish qualification on another OS or architecture. Signed managed bundles
retain the exact host-qualified wheels and remain the supported installation
route for the qualified target described in `docs/installation.md`.

## Public implementation coverage

`rust/oel-orbit/src/` contains the shared implementations for ONP integrators
and force models, planetary ONP and external harmonic gravity, OGP-SGP4/SDP4, CR3BP, relative dynamics, attitude, estimation,
control and actuation, environment and frames, coverage and free-space links,
spacecraft and vehicle geometry, and public standalone analysis kernels.
`rust/oel-game/src/` contains the Trainer's numerical, boundary, frame,
prediction and scoring owners. Existing Python bindings select these owners;
explicit Python backend selection retains reference execution.

The native source does not change public workflow availability. It does not
publish private campaign orchestration, operational Scale workflows,
customer inputs, private qualification fixtures, or external licensed datasets.
The public precision-force crate remains a source-level library entry point
sharing the same numeric owners rather than duplicating their implementations.

For the new planetary and external gravity routes, also run:

```bash
python -m pytest sim/tests/test_additional_gravity.py sim/tests/test_planetary_onp.py
```
