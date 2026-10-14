# Configuration And Python API Architecture

Wave 3 keeps OEL's established import contracts while moving their
implementations into focused modules. Existing callers should continue to use
`sim.config`, `sim.config.scenario_yaml`, `sim.api`, or the lazy public exports
from `sim`. The implementation packages are navigation surfaces for
maintainers and coding agents, not replacement public APIs.

## Scenario Configuration

`sim.config.scenario_yaml` remains the compatibility façade. Ownership under
`sim.config.scenario` is divided into schema models, parsing primitives,
preset resolution, object parsing, simulator parsing, analysis parsing,
output parsing, path security, cross-section validation, and loading.

The static `SCENARIO_CONFIG_FAMILIES` map in
`sim.config.scenario.architecture` identifies the implementation owner for
representative capabilities. Configuration normalization, object ordering,
legacy API conveniences, exception text, and path-policy behavior remain part
of the façade contract.

## Public Python API

`sim.api` remains the stable façade. `sim.public_api` separates configuration
wrappers, snapshots, results, sessions, workspaces, controller adapters, and
private-feature routing. The seven compatibility façade classes covered by the
architecture identity tests retain `sim.api` as their `__module__`, so repr,
pickle lookup, documented imports, and lazy exports from `sim` remain
compatible. Scenario authoring and validation classes such as
`ScenarioArtifact`, `ScenarioBuilder`, `ValidationIssue`, and
`ValidationReport` remain owned by `sim.scenarios` while being stable façade
exports.

The static `PUBLIC_API_FAMILIES` map in `sim.public_api.architecture` provides
the corresponding implementation index. Feature-routing remains centralized
and preserves the façade monkeypatch seam used by downstream tests and
integrations.

## Change Rules

- Add schema fields and parsing behavior to the owning scenario module, then
  re-export compatibility names through the façade when required.
- Add result queries to `results.py`, interactive lifecycle behavior to
  `session.py`, and higher-level workflows to `workspace.py`.
- Keep behavioral redesign separate from mechanical movement.
- Verify exact normalized configuration data and validation messages for
  configuration changes.
- Verify imports, signatures, class identity, repr, serialization, and
  representative run output parity for public API changes.


## Planetary trajectories

`sim.config.scenario.planetary` owns non-Earth normalization and compatibility validation; `sim.dynamics.planetary` owns prepared resources and the Rust adapter. Public snapshots retain an optional body-neutral state descriptor. See [Planetary ONP](planetary-onp.md).

`sim.config.scenario.additional_gravity` owns optional external harmonic sources
for Earth and planetary origins. Resource preparation lives in
`sim.dynamics.planetary.additional_gravity`; omitted selections retain existing
Earth serialization and normalized digests.
