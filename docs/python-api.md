# Python API

Use the Python API when you want to load scenarios, run simulations, inspect
payloads, or build analysis scripts without shelling out to the CLI.

The [OEL 1.x compatibility policy](compatibility-policy.md) defines the supported
imports, additive changes, deprecation window, and experimental API boundary.

The API follows the same scenario YAML contract as the CLI. Only run scenario
YAML from sources you trust; plugin pointers can import Python modules.

## Workspace Facade

Use `SimulationWorkspace` when you want CLI-style workflows from Python.

`SimulationWorkspace` and `SimulationSession` are trusted-local extension
surfaces: plugin validation may import configured Python modules. Hosted or
untrusted callers must use `HostedSimulationWorkspace` or
`HostedSimulationSession`. Those facades enforce sealed mode and structural
validation does not import plugin modules:

```python
from sim import HostedSimulationWorkspace

workspace = HostedSimulationWorkspace(workspace_root="/srv/oel/workspace")
report = workspace.validate(untrusted_config)
if not report["ok"]:
    raise ValueError(report["errors"])
result = workspace.run(untrusted_config)
```

For an agent-authored candidate, use
`SimulationWorkspace.validate_candidate_config(...)`. It always performs safe
structural/plugin-pointer validation first. It imports referenced plugins only
when `trust_plugins=True`, and it never treats validation as permission to run:

```python
candidate = workspace.validate_candidate_config("configs/agent_candidate.yaml")
if candidate["status"] == "safe_only":
    print(candidate["next_step"])

# Only after reviewing and trusting referenced modules and external paths:
trusted = workspace.validate_candidate_config(
    "configs/agent_candidate.yaml",
    trust_plugins=True,
)
```

For a lower-level structural-only check in trusted local tooling, call
`SimulationWorkspace.validate_safe(...)` before ordinary importing validation.

```python
from sim import SimulationWorkspace

workspace = SimulationWorkspace()

validation = workspace.validate("configs/quickstart_5min.yaml")
if not validation["ok"]:
    raise RuntimeError(validation["errors"])

result = workspace.run("configs/quickstart_5min.yaml")
print(result.summary["scenario_name"])
```

The v1 local edition also provides bounded Controller Bench, campaign, and
saved-output report workflows. Those standalone routes remain separate from
the scenario workspace façade and keep their own contracts and evidence;
Hosted service orchestration and provider-driven reports remain private.

## Bounded constellation-design API

Use `sim.constellation_design` to validate, evaluate, publish, or replay a
small explicit Walker/shell and ground-network trade:

```python
import json
from pathlib import Path

from sim.constellation_design import (
    ConstellationDesignProblem,
    solve_constellation_design,
    verify_constellation_design_artifacts,
    write_constellation_design_artifacts,
)

payload = json.loads(Path("my_constellation_trade.json").read_text())
problem = ConstellationDesignProblem.from_mapping(payload)
result = solve_constellation_design(problem)
artifacts = write_constellation_design_artifacts(result, "outputs/my_trade")
verified = verify_constellation_design_artifacts(artifacts.output_dir)
```

See [Constellation And Ground-Network Design](constellation-design.md) for the
resource envelope, evidence semantics, validation plan, and local/private-
service boundary.

## Scenario Artifacts

Use `ScenarioArtifact` when Python, notebooks, apps, or agents need to assemble
or mutate a scenario while keeping YAML as the durable review artifact.

```python
from sim import ScenarioArtifact, SimulationWorkspace

workspace = SimulationWorkspace()

artifact = ScenarioArtifact.from_dict({
    "scenario_name": "api_artifact_demo",
    "objects": {
        "target": {
            "enabled": True,
            "kind": "satellite",
            "specs": {"mass_kg": 100.0},
            "initial_state": {
                "position_eci_km": [7000.0, 0.0, 0.0],
                "velocity_eci_km_s": [0.0, 7.5, 0.0],
            },
        }
    },
    "simulator": {
        "duration_s": 120.0,
        "dt_s": 10.0,
        "dynamics": {"attitude": {"enabled": False}},
    },
    "outputs": {
        "output_dir": "outputs/api_artifact_demo",
        "mode": "save",
        "review": {"enabled": True, "detail": "standard"},
        "plots": {"enabled": False},
        "animations": {"enabled": False},
    },
})

report = artifact.validate_report(workspace)
if not report.ok:
    raise RuntimeError([issue.to_dict() for issue in report.errors])

workspace.save_config(artifact, "configs/generated/api_artifact_demo.yaml")
result = artifact.run(workspace)
manifest = result.evidence_manifest()
review = result.review()

print(manifest["review"]["db_path"])
print(review.query("SELECT scenario_name, samples FROM run_metadata").rows)
```

`ScenarioArtifact.to_yaml_text()` and `ScenarioArtifact.write(...)` use the same
schema-backed normalization path as ordinary config loading, then render a
clean artifact shape that omits compatibility/default noise such as disabled
legacy object stubs. The written YAML is intended to be inspected, committed,
diffed, and rerun through the CLI.

Use `ScenarioArtifact.to_dict()` when you need the fully normalized engine
configuration. Use `ScenarioArtifact.to_artifact_dict()` when you need the
reviewable YAML artifact shape.

`SimulationWorkspace.validate(...)` still returns the legacy dictionary shape.
Use `SimulationWorkspace.validate_report(...)` or
`ScenarioArtifact.validate_report(...)` when callers need structured
`ValidationIssue` objects for agent repair loops or UI display.

These authoring helpers live in `sim.scenarios` and are re-exported from
`sim` and `sim.api` for compatibility.

For repair loops on invalid draft dictionaries, call
`SimulationWorkspace.validate_report(...)` before creating a `ScenarioArtifact`.
`ScenarioArtifact.from_dict(...)` normalizes through the schema immediately and
may raise for invalid drafts.

```python
draft = artifact.to_dict()
draft["outputs"]["review"]["detail"] = "verbose"

report = workspace.validate_report(draft)
for issue in report.errors:
    print(issue.path)
    print(issue.message)
    print(issue.hint)
    print(issue.allowed_values)
```

Common validation issues include structured paths for timing fields,
`outputs.mode`, `outputs.review.detail`, ground-station fields, relative RIC
initial states, and plugin pointers.

## Scenario Builder

Use `ScenarioBuilder` for common authoring flows where a domain-facing Python
helper is clearer than assembling raw dictionaries. The builder remains
artifact-first: `artifact()` normalizes through the scenario YAML parser and
returns a `ScenarioArtifact`.

```python
from sim import ScenarioBuilder

artifact = (
    ScenarioBuilder("builder_rpo_demo")
    .duration(7200.0, dt_s=10.0)
    .target_satellite(
        mass_kg=100.0,
        position_eci_km=[7000.0, 0.0, 0.0],
        velocity_eci_km_s=[0.0, 7.5, 0.0],
    )
    .chaser_relative_ric(
        [0.0, -5.0, 0.0, 0.0, 0.0, 0.0],
        mass_kg=50.0,
        frame="rect",
    )
    .outputs("outputs/builder_rpo_demo")
    .review(detail="standard")
    .artifact()
)

artifact.write("configs/generated/builder_rpo_demo.yaml")
result = artifact.run()
print(result.primary_pair)
```

The first builder surface is intentionally small. It supports single-satellite
propagation and basic target/chaser RIC setup through helpers such as
`duration(...)`, `outputs(...)`, `review(...)`, `satellite(...)`,
`target_satellite(...)`, `chaser_relative_ric(...)`, and `ground_station(...)`.
For advanced scenario features, use `ScenarioArtifact.from_dict(...)` or
ordinary YAML until a domain-facing helper exists.

## Custom Metrics

`SimulationResult` exposes helpers for custom analysis that is too specific for
built-in simulator metrics.

```python
import numpy as np

from sim import SimulationWorkspace

workspace = SimulationWorkspace()
result = workspace.run("configs/my_deploy_case.yaml")

ric = result.relative_state("deployed_sat", "carrier_sat", frame="ric_rect")
range_after_10_min = result.range_between(
    "deployed_sat",
    "carrier_sat",
    start_s=600.0,
)

print(ric[0])
print(float(np.min(range_after_10_min)))
```

For one-off analysis, pass metric callbacks:

```python
def collision_metric(result):
    ranges = result.range_between("deployed_sat", "carrier_sat", start_s=600.0)
    min_range_km = float(np.min(ranges))
    return {
        "min_range_after_10_min_km": min_range_km,
        "collision_after_10_min": bool(min_range_km < 0.01),
    }

metrics = result.evaluate_metrics([collision_metric])
print(metrics)
```

Numeric custom metrics are summarized with count, mean, min, max, and
percentiles. Boolean metrics are summarized as true/false counts and
`probability_true` by Pro campaign helpers. In the public core,
`result.evaluate_metrics(...)` returns the per-run callback outputs directly.

Common event helpers are available directly on `SimulationResult`:

```python
pair = result.primary_pair
print(result.object_ids)
print(result.reference_object_id)

min_range_m = result.min_range("deployed_sat", "carrier_sat", units="m")
t_ca_s = result.time_of_min_range("deployed_sat", "carrier_sat")

collision = result.collision_event(
    "deployed_sat",
    "carrier_sat",
    radius_km=0.01,
    start_s=600.0,
)

violations = result.keepout_violations(
    "deployed_sat",
    "carrier_sat",
    radius_km=0.1,
)
```

## DataFrames And Records

Time histories can be exported as record lists. If `pandas` is installed, the
`to_dataframe` helpers return a DataFrame; otherwise they return the same
record-list shape.

```python
truth = result.to_records("truth", object_id="deployed_sat")
truth_df = result.to_dataframe("truth", object_id="deployed_sat")

relative_df = result.relative_dataframe(
    "deployed_sat",
    "carrier_sat",
    frame="ric_rect",
)
```

## Config Mutations

Use `SimulationConfig` mutation helpers for small what-if studies:

```python
cfg = workspace.load("configs/my_deploy_case.yaml")
fast = cfg.with_value(
    "deployed_sat.initial_state.relative_to_target_ric.state[4]",
    0.001,
)
fast = fast.with_output_dir("outputs/deploy_fast")
result = workspace.run(fast)
```

For a complete public script using this API surface, run:

```bash
.venv/bin/python examples/python/flagship_analysis.py
```

It runs `configs/ric_pd_10km_experiment.yaml` and writes a custom metrics JSON
and CSV package under `outputs/flagship_ric_pd_10km/custom_analysis/`.

## Run A Scenario

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)
result = session.run()

print(result.summary["scenario_name"])
print(result.summary["samples"])
```

The `SimulationResult` wraps the run payload and exposes the most common
fields as properties.

```python
print(result.time_s[:5])
print(result.truth.keys())
print(result.applied_thrust.keys())
```

## Inspect A Snapshot

For single-run scenarios, snapshots provide one indexed view of truth, belief,
and applied commands.

For v2 satellites, `belief` is the latest navigation estimate published by the
stack. It is distinct from raw sensor measurements. Unavailable components are
NaN; a stack with no measurement shape yet may have an empty vector. Read
[flight-software observations](flight-software-observations.md) for the frame,
epoch, freshness and raw-packet inspection contracts.

```python
snap = result.snapshot(0)

print(snap.time_s)
print(snap.truth.keys())
```

Snapshot fields are useful for notebooks and small scripts. For reusable tools,
prefer summary fields and artifact maps documented in the payload contract.

## Step A Session

Use stepping when you need interactive control, custom loop logic, or live
inspection.

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/quickstart_5min.yaml")
session = SimulationSession.from_config(cfg)

snap = session.reset()
while not session.done:
    snap = session.step()
    print(snap.step_index, snap.time_s)
```

Calling `step()` after completion returns the final snapshot.

## Ground-Station Access

When a scenario defines `ground_stations`, the result exposes passive access
histories.

The public TLE access example initializes the object from TLE lines, then runs
ONP, the OEL Numerical Propagator. It is not an OGP-SGP4/general-perturbations
access workflow; use an object with `propagation_method: general` and
`general.model: sgp4` for passive OGP-SGP4 propagation.

```python
cfg = SimulationConfig.from_yaml(
    "examples/configs/public_ground_station_access_from_tle.yaml"
)
result = SimulationSession.from_config(cfg).run()

summary = result.summary["ground_station_access_summary"]
for station_id, station_summary in summary.items():
    print(station_id, station_summary)

for station_id, station_payload in result.ground_station_access.items():
    for object_id, access in station_payload["targets"].items():
        print(station_id, object_id, access["access"][:5])
```

Single-run artifacts also include two Markdown access reports when access data
exists: one organized by satellite and one organized by ground station. Report
AOS/LOS times are UTC based on `simulator.initial_jd_utc`, defaulting to
`2026-01-01T00:00:00Z` when no epoch is configured.

Ground-station access is passive. It does not modify truth, belief, control,
knowledge, or termination.

Add `ground_station_access` to `outputs.plots.figure_ids` for the built-in
access/elevation/range figure, and set `outputs.plots.draw_earth_map: true`
when static ground tracks should use a world-map background.

## Ground and spacecraft resource snapshots

The ground-segment demo enables both layers for object `sat`. Load and run it
to inspect both from one snapshot:

```python
from sim import SimulationConfig, SimulationSession

cfg = SimulationConfig.from_yaml("configs/ground_segment_demo.yaml")
result = SimulationSession.from_config(cfg).run()
snap = result.snapshot(10)
print(snap.ground_segment.get("objects", {}).get("sat"))
print(snap.spacecraft_resources.get("sat"))
```

`ground_segment` contains received packets, station service status, retained
telemetry, and the ground-side orbit posterior/prediction. The
`spacecraft_resources` mapping contains simulation-truth temperature and
electrical-resource values. These truth values are not onboard measurements or
ground knowledge; only the separately modeled ground segment receives its
configured ideal telemetry packets. See [Ground Segment](models/ground-segment.md)
and [Spacecraft Resources](models/spacecraft-resources.md) for the model limits.

## Coverage And Directed-Link Analysis

The public analysis façade exposes deterministic post-processing kernels and
history adapters:

```python
from sim.analysis import (
    DirectedLinkConfig,
    GlobalCoverageConfig,
    evaluate_history_directed_link,
    evaluate_history_global_coverage,
    history_from_single_run,
)
```

For the integrated scenario workflow, enable `outputs.orbital_analysis` and run
the scenario normally. The result summary records the number and schema version
of the analyses, content-bound artifacts are written beneath
`orbital_analysis/`, and an enabled review store receives `coverage_*` and
`link_*` tables. Start with
`examples/configs/public_coverage_and_link_analysis.yaml` and
[Coverage And Link Scenario Analysis](coverage-link-scenario-analysis.md).

These analyses consume deterministic evidence after propagation. They do not
alter the simulation unless a caller separately configures the bounded,
authorized directed-link runtime monitor.

## Artifacts

Output artifacts are recorded in the result summary. Use artifact maps instead
of constructing filenames from assumptions.

```python
result = SimulationSession.from_config(
    SimulationConfig.from_yaml("configs/plotting_rendezvous_demo.yaml")
).run()

print(result.summary["output_index_md"])
print(result.summary["plot_outputs"])
```

If you need arbitrary time-series analysis, configure:

```yaml
outputs:
  stats:
    save_json: true
    save_full_log: true
```

Then load `master_run_log.json` from the output directory.

## Lower-Level Execution Helper

For scripts that want the raw payload directly:

```python
from sim.execution import run_simulation_config_file

payload = run_simulation_config_file("configs/quickstart_5min.yaml")
print(payload["run"]["scenario_name"])
```

The `SimulationSession` wrapper is the preferred public API when you need a
stable object-oriented surface.

### Session Control And FSW Inputs

For trusted interactive single-run work, `SimulationSession` exposes
`publish_fsw_input(object_id, event)`,
`add_fsw_input_publisher(object_id, publisher)`, and
`request_fsw_input_publisher_poll(object_id)`. Publishers are sampled only at
flight-software release points; an explicit poll request schedules one poll at
the current simulation time.

Trusted sessions also expose object/factory pairs for orbit and attitude
controllers, mission strategy, and mission execution:
`set_orbit_controller[_factory]`, `set_attitude_controller[_factory]`,
`set_mission_strategy[_factory]`, and `set_mission_execution[_factory]`.
Factories create fresh stateful instances after an engine reset. Sealed
sessions reject Python runtime overrides. GNC-v2 satellites reject individual
controller/mission overrides and require one complete
`objects.<id>.flight_software` stack in scenario YAML.

See [Plotting](plotting.md) for the current single-run figure ID catalog and
the public plotting helper functions.

## Extension Guidance

Supported extension paths:

- scenario YAML
- object presets
- importable plugin pointers
- public controller or mission interfaces
- `SimulationSession`

Avoid importing private helpers or classes whose names start with `_`. Those
are implementation details and may change while the project is pre-1.0.

Related docs:

- [Scenario YAML](scenario-yaml.md)
- [Payload And Artifact Contract](contracts/payload-artifact-contract.md)
- [Engine Contract](contracts/engine-contract.md)
