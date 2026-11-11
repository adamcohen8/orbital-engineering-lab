# Parallel OGP Within One Simulation

Public OEL supports bounded Rust OGP-SGP4/SDP4 propagation threads inside one
deterministic simulation. The owner still visits every sample, applies results
in object order, and handles callbacks, events, termination, review and outputs.
This optional backend uses native object-row threads for passive OGP history
preparation. ONP object workers and hierarchical local campaign budgets use
separate execution controls in [Scenario YAML](scenario-yaml.md). See
[OEL v1 local workflows](v1-local-workflows.md) for campaign, Monte Carlo,
sensitivity, and config-queue entry points. Hosted and Manifold orchestration
have separate service contracts.

## Build The Optional Public Backend

From a source checkout with the environment in `installation.md` activated,
install a Rust toolchain and Maturin, then build the OGP-only extension:

```bash
python -m pip install 'maturin>=1.12,<2'
python -m maturin build --release --locked --features python \
  --manifest-path rust/oel-ogp/Cargo.toml --out dist/ogp
python -m pip install dist/ogp/oel_rust_ogp-*.whl
```

PowerShell users can pass the wheel's exact path to the final command. Initial
builds may fetch the locked Cargo dependencies. For offline builds, cache those
dependencies and the build tools first, then add `--offline`. This optional
extension is separate from the standard pure-Python install.

The public crate compiles the same authoritative `rust/oel-orbit/src/ogp.rs`
and shared Python bindings as the full native build. It includes the pinned
MIT-licensed SGP4 dependency, its license, patch notes and upstream hashes. It
exposes only scalar and batch OGP propagation. It does not expose ONP kernels,
OD observation fitting, Scale screening or campaign entry points. The existing
full native core keeps its reviewed public source surface. A compatible
installed `oel_rust_orbit` wheel also works.

## Configure And Run

Set `general.numeric_backend: rust` on each enabled OGP object, then add:

```yaml
simulator:
  execution:
    ogp_parallelism:
      enabled: true
      workers: 2
```

`workers: 0` chooses a bounded automatic allocation (at most four workers,
reserving one logical CPU). Explicit counts are capped by available CPUs,
the resource profile and any enclosing campaign budget. `policy: serial`
and the `laptop-safe` profile use one worker. Use `standard` for the bounded
two-worker example. General `object_parallelism` and `policy: auto/parallel`
remain separate object-worker controls and cannot be combined with this option.

The synthetic 16-object mixed LEO/GEO example explicitly selects Python for
other numerical subsystems, so it needs only the OGP extension even when the
installed OEL version defaults other kernels to Rust. Validate and run it:

```bash
python run_simulation.py --config examples/configs/public_ogp_parallel.yaml --validate-only
python run_simulation.py --config examples/configs/public_ogp_parallel.yaml
```

Inspect `outputs/public_ogp_parallel/master_run_summary.json` and
`outputs/public_ogp_parallel/review/run.sqlite`. The runtime planner records
`parallel_scope: ogp_history_preparation`, actual `selected_backend` and peak
`selected_workers`, plus the latest batch's native allocation/use metadata.
A worker request alone does not establish actual parallel execution.

## Limits And Validation

Each preparation group contains at most 64 objects and 257 epochs per object.
The native kernel keeps fewer than 4096 object/time samples or a single-object
history serial. Short runs can therefore legitimately remain serial. Mixed
scenarios retain serial ONP steps. Coordinated collision/system-force stepping
does not use this optimization. State or cadence changes invalidate cached rows;
speculative future propagation errors return to ordinary ordered stepping.

The public regression tests compare complete mixed SGP4/SDP4 scenario histories
and callback order against serial Rust OGP, check resource/serial policy, and
verify that ONP workers still require Pro. The native core tests compare exact
serial/parallel arrays and ordered errors, including resonance query resets:

```bash
python -m pytest -q sim/tests/test_public_ogp_parallelism.py
cargo test --locked --manifest-path rust/oel-ogp/Cargo.toml
```

These checks establish deterministic execution parity within their fixtures.
They do not establish empirical orbit accuracy, operational readiness, or a
speedup for every scenario. Python OGP remains available for serial runs.
