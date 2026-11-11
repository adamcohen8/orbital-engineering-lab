# Plotting

Orbital Engineering Lab includes public single-run plotting for quick inspection,
debugging, and documentation artifacts. Campaign, benchmark, optimization, and
sensitivity plots are part of the local v1 workflow layer.

## YAML Usage

```yaml
outputs:
  output_dir: "outputs/my_run"
  mode: "save"
  plots:
    enabled: true
    preset: "rendezvous"
    reference_object_id: "target"
    draw_earth_map: true
    style: "oel_dark"
    figure_ids:
      - "run_dashboard"
      - "rendezvous_summary"
    dpi: 160
  animations:
    enabled: false
    style: "oel_dark"
```

`preset` expands to a useful bundle of figure IDs. You can add more
`figure_ids` alongside a preset. `reference_object_id` should name an active
object from the scenario's canonical `objects` map; `target` is only the
convention used by the rendezvous examples.
Set `orbital_elements_object_id` under `outputs.plots` when you want COE figures
to focus on one object; otherwise they overlay all objects with valid state
histories.
Set `draw_earth_map: true` under `outputs.plots` to use a world-map background
for static `ground_track` and `ground_track_multi` figures. The plotter uses
Cartopy when available and otherwise falls back to a lightweight built-in map.
Set `style` to `oel_dark` for the default screen/game-aligned OEL artifact
identity, `oel_light` for print-friendly report figures, or `matplotlib` to use
unbranded Matplotlib defaults. Saved animations use the same visual identity
and footer metadata; set `outputs.animations.style` only when a movie should
override `outputs.plots.style`. Branded static plots share the same save path,
footer metadata, role-color conventions, and show/close behavior across
single-run, campaign, benchmark, game-debrief, validation, and animation
workflows.

Agent-native review plots additionally apply the versioned
[OEL Plot Quality Contract](plot-quality-contract.md). It selects one stable
numeric formatter per axis, checks rendered text and legend geometry, applies
bounded presentation-only repairs, records the resulting quality receipt, and
still requires agent visual inspection before handoff.

Supported saved review animations apply the separate
[OEL Animation Quality Contract](animation-quality-contract.md). It freezes
sequence-wide numeric formatting and declared camera behavior, checks every
frame, verifies the encoded artifact, and produces a quality receipt plus a
contact sheet for mandatory agent visual review.

For the flagship plotting scenario used by the checked-in gallery:

```bash
.venv/bin/python run_simulation.py --config configs/ric_pd_10km_experiment.yaml
```

See the [plot gallery](plot-gallery.md) for checked-in examples generated from
that config.

## Plot Presets

- `minimal`: run dashboard
- `orbit`: run dashboard, multi-object ECI trajectory, multi-object ground track, orbital-elements summary
- `rendezvous`: run dashboard, rendezvous summary, RIC projections, relative range, control effort
- `attitude`: run dashboard, quaternion components, body rates, quaternion error
- `estimation`: estimation error norms, component errors, knowledge timeline, sensor access, truth/measurement/estimate filtering
- `access`: ground-station access and multi-object ground track
- `rocket`: run dashboard, ascent/GNC diagnostics, fuel, orbital elements, timeline, downrange/altitude, max-Q throttle, TVC/aero authority, and insertion scorecard
- `reentry`: atmospheric re-entry summary, aero, and thermal-load plots
- `aero_assist`: atmospheric-pass summary, lift/drag histories, thermal-load, lift-axis alignment, and ECI trajectory plots
- `debug`: every public single-run figure ID

## Common Figure IDs

- `run_dashboard`: one-page summary of trajectory, relative motion, thrust, delta-v, and rates
- `rendezvous_summary`: RIC projections, range, relative speed, and RIC components
- `rendezvous_summary_curvilinear`: curvilinear RIC projections, combined relative range/speed with two y axes, curvilinear RIC components, and cumulative target/chaser delta-v
- `orbital_element_a`: semi-major axis over time
- `orbital_element_ecc`: eccentricity over time
- `orbital_element_inc`: inclination over time
- `orbital_element_raan`: right ascension of ascending node over time
- `orbital_element_argp`: argument of perigee over time
- `orbital_element_true_anomaly`: true anomaly over time
- `orbital_elements_summary`: six-panel COE history, arranged as two columns by three rows
- `orbital_elements_angles`: inclination, RAAN, argument of perigee, and true anomaly on one plot
- `ground_track`: per-object static ground track
- `ground_track_multi`: all-object static ground track
- `control_effort`: thrust components, magnitude, and cumulative delta-v
- `estimation_error`: position and velocity belief error against truth
- `estimation_error_components`: position and velocity component errors
- `knowledge_filtering`: truth, raw measurement, and filtered estimate comparison for knowledge tracks, including position/velocity residual histograms and normalized residuals when sensor noise is supplied
- `sensor_access`: observer-target access timeline, range, and knowledge position error
- `ground_station_access`: station-target access timeline, elevation, and slant range
- `attitude_control_summary`: quaternion tracking error, body-rate norm, thrust magnitude, and thrust-alignment error
- `rocket_ascent_diagnostics`: altitude, speed, dynamic pressure, mass, throttle, and key ascent telemetry
- `rocket_gnc_diagnostics`: pitch/yaw/roll guidance and control diagnostics
- `rocket_orbital_elements`: rocket orbital-element history
- `rocket_fuel_remaining`: remaining propellant history
- `rocket_mission_timeline`: liftoff, pitch-program, max-Q, stage-event, insertion-band, and final-sample timing
- `rocket_downrange_altitude`: launch altitude versus downrange distance, colored by speed when available
- `rocket_maxq_throttle`: dynamic pressure, throttle, Mach, and altitude around max-Q limiting
- `rocket_tvc_aero_authority`: TVC gimbal, alpha/beta, aero loads, dynamic pressure, and thrust-to-weight
- `rocket_insertion_scorecard`: final orbit/resource/control scorecard against configured insertion targets
- `reentry_summary`: altitude, dynamic pressure, g-load, and heat-rate overview for tracked re-entry objects
- `reentry_aero`: density, relative atmospheric speed, dynamic pressure, and drag-deceleration histories
- `reentry_thermal`: Sutton-Graves heat-rate and integrated heat-load histories
- `atmospheric_pass`: altitude, drag/lift acceleration, dynamic pressure, cross-track motion, heat load, and lift-axis alignment
- `trajectory_eci_multi`: all-object 3D ECI trajectories
- `trajectory_ric_curv_2d_multi`: all-object RIC curvilinear 2D projections
- `trajectory_ric_curv_2d_multi_target_burns`: all-object RIC curvilinear 2D projections with orange dots at target-burn time samples
- `trajectory_ric_rect_2d_multi_target_burns`: all-object rectangular-RIC 2D projections with orange dots at target-burn time samples
- `relative_range`: pairwise relative range over time
- `control_thrust`: per-object thrust component history

The full list is available in `sim.master_outputs.AVAILABLE_FIGURE_IDS`.

## Implementation Map

`sim.master_outputs` is the stable façade for figure IDs, presets,
`plot_outputs`, and `animate_outputs`. The implementation is split by
capability under `sim.plotting`:

- `output_registry.py`: ordered renderer-family registry and figure ownership
- `summary_outputs.py`: dashboards, orbit/access/estimation summaries, and re-entry plots
- `trajectory_outputs.py`: frame trajectories, RIC projections, quaternions, and body rates
- `rocket_outputs.py`: ascent, GNC, fuel, mission-timeline, and insertion plots
- `control_outputs.py`: thrust histories, delta-v remaining, and alignment error
- `knowledge_outputs.py`: knowledge timelines
- `output_animations.py`: single-run animations
- `animation_quality.py`: versioned per-frame, temporal, encoding, receipt, and contact-sheet QA
- `output_context.py` and `output_helpers.py`: shared run context and numerical helpers

The public single-run plotting functions exposed through `sim.plotting` and
`sim.plotting.single_run` are implemented in these focused families:

- `dashboard_plots.py`: run dashboards, rendezvous summaries, and control effort
- `estimation_plots.py`: estimation, knowledge-filtering, and sensor-access plots
- `access_plots.py`: payload ground tracks and ground-station access
- `orbital_element_plots.py`: orbital-element plots and summaries
- `reentry_plots.py`: atmospheric-pass, aerodynamic, thermal, and re-entry summaries
- `attitude_summary_plots.py`: attitude-control summary
- `single_run_context.py` and `single_run_math.py`: shared payload and numerical helpers

The lower-level compatibility API in `sim.utils.plotting_capabilities` is
implemented by `frame_plots.py`, `control_plots.py`, `ground_track_plots.py`,
`trajectory_animations.py`, `attitude_animations.py`, and
`prism_animations.py`. Shared animation runtime behavior and geometry live in
`capability_common.py` and `attitude_geometry.py`.

Start with `PLOT_RENDERER_FAMILIES` in `output_registry.py` when locating the
owner of a figure ID. Use `PLOTTING_CAPABILITY_FAMILIES` in
`architecture.py` to locate the implementation owner of a public plotting
function. External callers should continue to use the stable façades.

For plots that are not built in, see [Custom Analysis](custom-analysis.md) for
examples that load saved JSON/CSV artifacts and create Matplotlib figures.
Ground-station access is also saved as payload data and summary metrics, so you
can use the custom-analysis path when you need a station-specific or
publication-specific variant.

The target-burn RIC variants mark `target` burns by default. Set
`outputs.plots.burn_marker_object_ids` to a list of object IDs when you want the
same orange markers driven by a different burn source.

## Python API

The `sim.plotting` package works from a single-run payload:

```python
from sim.execution import run_simulation_config_file
from sim.plotting import plot_run_dashboard, plot_rendezvous_summary

payload = run_simulation_config_file("configs/hcw_lqr_two_body_perfect.yaml")

fig = plot_run_dashboard(payload, out_path="outputs/dashboard.png", close=True)
fig = plot_rendezvous_summary(payload, out_path="outputs/rendezvous.png", close=True)
```

Available API functions:

- `plot_run_dashboard`
- `plot_rendezvous_summary`
- `plot_control_effort`
- `plot_estimation_error`
- `plot_estimation_error_components`
- `plot_knowledge_filtering`
- `plot_atmospheric_pass`
- `plot_ground_track_from_payload`
- `plot_ground_station_access`
- `plot_orbital_element`
- `plot_orbital_elements_summary`
- `plot_orbital_elements_angles`
- `plot_attitude_control_summary`
- `plot_sensor_access`

Custom Matplotlib figures should use `sim.plotting.style.save_oel_figure` when
they are intended to sit beside OEL-generated artifacts. That keeps footer
metadata, theme colors, output directory creation, and save behavior consistent
with built-in plots.

## Local Edition And Private-Service Boundary

Public plotting focuses on understanding one run. Local workflow plotting
supports many-run summaries such as Monte Carlo histograms, sensitivity plots,
controller-benchmark comparisons, optimization convergence, campaign
dashboards, baseline comparison, and report packs. Hosted service
infrastructure, provider-driven report generation, and the Manifold remain
private.

Monte Carlo campaigns with generic metric gates also emit study-metric
histograms, CDFs, sampled-parameter versus metric scatter plots, pass/fail
parameter maps, and metric-gate margin plots. Sensitivity studies emit
method-specific response, scatter, or heatmap plots plus ranking bars and, when
a numeric baseline is available, baseline-delta bars.
