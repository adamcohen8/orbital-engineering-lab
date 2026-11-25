# Coverage and Link Scenario Analysis

OEL can run evidence-only whole-Earth conical coverage and directed
spacecraft-to-spacecraft or spacecraft-to-ground-station link analysis after a
single ONP or scenario OGP propagation.
This post-processing does not affect commands, modes, or subsequent dynamics.
The deterministic implementation, adapters, contracts, tests, and this
workflow are part of the public core with an experimental support posture.

The scenario must declare an absolute epoch. Directional coverage and
directional terminals require achieved attitude dynamics; OEL fails closed
when that evidence is unavailable. A constant-gain terminal is attitude
independent. OGP products never acquire implied attitude.

Public Scenario YAML with `propagation_method: general` supports only
attitude-independent orbital-analysis endpoints because scenario OGP retains
no achieved attitude history. Directional sensors or terminals must use ONP
with achieved attitude, or a programmatic ECI OGP history paired explicitly
with replay or analytic-ideal attitude. Static initial OGP attitude is never
treated as achieved attitude.

```yaml
simulator:
  initial_jd_utc: 2461041.5

outputs:
  review:
    enabled: true
  orbital_analysis:
    enabled: true
    coverage:
      - analysis_id: satellite_a_global
        source_object_id: satellite_a
        sensor_id: satellite_a.imager
        order: 5
        half_angle_deg: 20.0
        quat_body_from_sensor: [1.0, 0.0, 0.0, 0.0]
        transition_time_tolerance_s: 0.1
        transition_max_iterations: 50
        include_fraction_plot: true
    directed_links:
      - analysis_id: satellite_a_to_b
        link_id: satellite_a_to_b
        tx_object_id: satellite_a
        rx_object_id: satellite_b
        tx_terminal:
          terminal_id: satellite_a.tx
          pattern: {kind: constant, gain_dbi: 6.0}
        rx_terminal:
          terminal_id: satellite_b.rx
          pattern: {kind: constant, gain_dbi: 6.0}
        carrier_frequency_hz: 2200000000.0
        tx_power_w: 10.0
        data_rate_bps: 1000000.0
        system_noise_temperature_k: 300.0
        required_eb_n0_db: 5.0
        transition_time_tolerance_s: 0.1
        transition_max_iterations: 50
        include_margin_plot: true
```

For a configured ground station, replace exactly one object selector with a
ground-station selector. The station's `min_elevation_deg` and `max_range_km`
become link defaults unless the link overrides them. Ground-station terminal
mounting is in the local ENU parent frame:

```yaml
ground_stations:
  - id: colorado_springs
    lat_deg: 38.8339
    lon_deg: -104.8214
    alt_km: 1.84
    min_elevation_deg: 10.0
    max_range_km: 2500.0

outputs:
  orbital_analysis:
    enabled: true
    directed_links:
      - analysis_id: satellite_a_to_colorado_springs
        link_id: satellite_a_to_colorado_springs
        tx_object_id: satellite_a
        rx_ground_station_id: colorado_springs
        tx_terminal:
          terminal_id: satellite_a.tx
          pattern: {kind: constant, gain_dbi: 6.0}
        rx_terminal:
          terminal_id: colorado_springs.rx
          quat_parent_from_terminal: [1.0, 0.0, 0.0, 0.0]
          pattern: {kind: constant, gain_dbi: 30.0}
        carrier_frequency_hz: 2200000000.0
        tx_power_w: 10.0
        data_rate_bps: 1000000.0
        system_noise_temperature_k: 300.0
        required_eb_n0_db: 5.0
```

Each endpoint must declare exactly one of `*_object_id` or
`*_ground_station_id`. Fixed-site-to-fixed-site links are outside this product.

Coverage artifacts are written under
`orbital_analysis/coverage/<analysis_id>/`; link artifacts are written under
`orbital_analysis/directed_links/<analysis_id>/`. When review output is
enabled, the queryable tables are:

- `coverage_summary`, `coverage_samples`, `coverage_intervals`, and
  `coverage_transitions`;
- `link_summary`, `link_samples`, `link_windows`, and `link_transitions`.

In link tables, the legacy column names `tx_object_id` and `rx_object_id` hold
generalized endpoint IDs; a fixed site therefore appears in an
`*_object_id` column. Use the adjacent `tx_endpoint_kind` and
`rx_endpoint_kind` columns to distinguish `spacecraft` from
`fixed_wgs84_site`. Terminal parent frames are not separate authoring fields
in scenario YAML: OEL resolves `body` for spacecraft endpoints and `enu` for
fixed-site endpoints, then retains those resolved values as
`tx_terminal_parent_frame` and `rx_terminal_parent_frame` provenance.

For example:

```sql
SELECT s.link_id, s.tx_object_id, s.rx_object_id,
       SUM(l.available) AS available_samples
FROM link_summary s LEFT JOIN link_samples l USING (analysis_id)
WHERE s.rx_object_id = 'colorado_springs'
GROUP BY s.analysis_id, s.link_id, s.tx_object_id, s.rx_object_id;
```

The standard run summary and `index.md` surface coverage and availability
results directly. Requested native plots are included in the ordinary plot
inventory and each has a sibling `.quality.json` receipt from OEL's strict
plot-quality policy.

Completed ONP/review histories and completed ECI OGP products normalize through
`sim.analysis.history_adapters.AnalysisHistory`. Retained state histories use
cubic Hermite position/velocity interpolation and shortest-arc quaternion
SLERP for event refinement. A caller may provide an arbitrary-epoch state
provider instead. The evidence records the refinement source; the two are not
presented as equivalent propagation claims.

This adapter currently covers the canonical conical sensor and directed-link
products. Rich footprints, communications coverage, constellation aggregation,
tasking, and runtime-causal consumers remain separate integration work.

Validate and run the checked-in public example with:

```bash
python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml --validate-only
python run_simulation.py --config examples/configs/public_coverage_and_link_analysis.yaml
python -m sim.agent_task run coverage_link_review --output-root outputs/agent_tasks
python -m sim.review outputs/examples/public_coverage_and_link_analysis --saved-query coverage_summary
python -m sim.review outputs/examples/public_coverage_and_link_analysis --saved-query directed_link_summary
```

When OEL MCP is connected, `coverage_link_review` is a supported public task
recipe. The MCP review surface also exposes the named queries
`coverage_summary`, `coverage_transition_summary`, `directed_link_summary`, and
`directed_link_windows`, plus the `coverage_fraction` and
`directed_link_margin` plot recipes.

Coverage and directed-link samples are evaluated on the retained scenario
history. Changing `simulator.dt_s` therefore changes propagation/history
sampling as well as analysis cadence; it is not a pure post-processing
resample. Window rows carry start/end censoring and transition/refinement
dispositions so a window spanning the study boundary is not over-read as a
fully observed acquisition or loss.

This workflow is deterministic engineering analysis, not calibrated sensor
performance or operational communications assurance. See the individual
contracts and [programmatic acceptance record](validation-coverage-link-programmatic.md)
for exact supported inputs, evidence, and non-claims.

## Access chains

An access chain finds intervals when all directed links on at least one path
meet their configured geometry, pointing, and RF constraints at the same epoch.
An optional sensor gate requires simultaneous visibility of a surface target
from the spacecraft at the start of the path. Each path uses the configured
link service rates and an explicit simultaneous receive/transmit assumption for
each spacecraft relay. Set `service_rate_bps` to the requested service rate.
Each leg must have been evaluated at that rate or higher. Declare every
intermediate spacecraft in `simultaneous_relays` only if its configured terminal
pair can operate together at that rate.

In scenario YAML, put this entry under `outputs.orbital_analysis.access_chains`
and enable `outputs.orbital_analysis`:

```yaml
- analysis_id: live_downlink
  service_rate_bps: 1000000
  simultaneous_relays: [satellite_b, satellite_c]
  sensor_gate:
    coverage_analysis_id: satellite_a_global
    latitude_deg: 90.0
    longitude_deg: 0.0
  paths:
    - path_id: via_relay_a
      links: [imager_to_relay_a, relay_a_to_north]
    - path_id: via_relay_b
      links: [imager_to_relay_b, relay_b_to_alternate]
```

The ordered `links` entries are **directed-link analysis IDs**. Use
`discover` in place of `paths` to enumerate cycle-free paths from
`source_object_id` to any `destination_ground_station_ids` through declared
relays. Set `max_hops` from 1 through 16. With `link_analysis_ids`, discovery
uses only the named links. Otherwise it searches all recorded links. Discovery
is bounded to 256 candidate paths and assigns IDs deterministically for the
same request. Explicit paths may also end at a spacecraft.

For a sensor gate, reference an existing conical `coverage_analysis_id`. Supply
either `cell_index` in its NESTED HEALPix grid or `latitude_deg` and
`longitude_deg` for a literal zero-height WGS84 surface point. A cell uses
the recorded cell-center intervals. A literal position uses the same cone
geometry and refinement settings with the source spacecraft's achieved state
and attitude evaluated at the requested point. Omit the gate for an RF-only
path analysis.

### Completed-run analysis

Run the same analysis after propagation with a JSON request. The completed run
needs a review store containing the directed-link evidence, plus the conical
coverage analysis and achieved attitude if the request uses a sensor gate. It
can have been created without any access-chain entry. The command reads the
store and writes derived artifacts to a new directory.

```bash
oel review access-chains outputs/examples/public_access_chains \
  --request-json '{
    "analysis_id":"live_downlink",
    "service_rate_bps":1000000,
    "simultaneous_relays":["satellite_b","satellite_c"],
    "sensor_gate":{"coverage_analysis_id":"satellite_a_global","latitude_deg":90,"longitude_deg":0},
    "discover":{
      "source_object_id":"satellite_a",
      "destination_ground_station_ids":["north_site","alternate_site"],
      "max_hops":2
    }
  }' \
  --exclude-ground-station north_site \
  --output-dir outputs/examples/public_access_chains_north_outage
```

Use `--request <file.json>` for saved requests. Repeat
`--exclude-ground-station` to remove multiple sites. The derived evidence
includes the service intervals lost relative to the same request with all
sites available.

### Review evidence

Open `access_chain_review.html` for the timeline and expand an acquisition
or loss time to inspect the controlling link or sensor condition.
`access_chain.json` records per-path windows, the union of service windows,
segments with the available path IDs, and the best path at each qualifying
retained epoch. Overlap contributes once to total availability. The ranking
maximizes the minimum RF margin along the path and breaks ties by path ID.
A change in sampled ranking establishes the preferred route at those samples.
Windows are half-open. Boundary reasons, refinement dispositions, and study
censoring come from the parent evidence. In particular, coarse sampling can
miss transitions. Set `include_timeline: false` to write JSON alone.

Scenario-configured chains also populate `access_chain_summary`,
`access_chain_windows`, `access_chain_segments`, and
`access_chain_ranked_samples` in the run's review store. Use
`oel review <run> --saved-query access_chain_windows` for boundary evidence.
These products describe modeled live-path feasibility with the declared
simultaneous-relay assumptions. They carry the limits of the input link
budgets and sensor geometry and make no packet delivery or schedule claim.

### Matched STK comparison

[The example](../examples/configs/public_access_chains.yaml) has one source,
two relays, two destinations, and an exact polar target. Use identical epoch
seconds, state and attitude histories, fixed sites, sensor cone, and per-leg
constraints for an end-to-end STK comparison. Match Earth and frame
conventions, RF assumptions, and event time tolerance. Export STK's
[Strand Access](https://help.agi.com/stk/Subsystems/dataProviders/Content/html/Chain.htm)
report and compare interval count and each start and end time by ordered
route against `paths[].windows`. Compare combined availability and the
station-outage result only after the per-route comparison.

To isolate interval composition from geometry differences, use STK's
[Intervals constraint](https://help.agi.com/stk/Content/stk/constraints-03.htm)
with the same pairwise windows on separate proxy objects for each leg of each
strand. In the known-answer case in `sim/tests/test_access_chains.py`, the
expected chain windows are 25–32 and 38–40 seconds via the first relay and
30–32 and 38–45 seconds via the second. A third proposed path has zero
duration at the single touching boundary. The service union is 25–32 and
38–45 seconds. Removing Denver loses 25–30 seconds. This fixture tests time
composition independently of orbit and RF calculations.
