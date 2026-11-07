# Ground segment

The optional ground segment maintains what a ground operator can know, separately
from simulation truth and onboard navigation. It works in headless simulations
and the step-driven Python API; it has no dependency on an Operator Trainer.

Enable the top-level `ground_segment` section. See
[`ground_segment_demo.yaml`](../../configs/ground_segment_demo.yaml) for a complete
1 Hz thermal/power and tracking example with a scheduled contact interruption.

```yaml
ground_segment:
  enabled: true
  cadence_s: 1.0
  latency_s: 2.0
  stale_after_s: 10.0
  prediction_step_s: 1.0
  process_noise_diag: [0, 0, 0, 0.00000001, 0.00000001, 0.00000001]
  priors: {}  # No prior means unknown orbit, even after receiving tracking.
  outages:
    - station_id: station
      start_s: 20
      end_s: 40
      services: [tracking, downlink, uplink]
```

`priors.<object_id>` accepts an explicit six-element `state` (ECI km and km/s),
a symmetric positive-semidefinite 6x6 `covariance` in corresponding squared/cross
units, and `epoch_s` relative to simulation start (zero or earlier). This is an
operator-supplied prior, not an initialization from simulation truth. A prior
is required for orbit estimation; initial orbit determination is not included.

## Data path

1. The simulation adapter uses existing ground-station line-of-sight, elevation,
   and range masks. Outages suppress tracking, downlink, and uplink independently.
2. Stations with `measurements.enabled` generate existing seeded noisy azimuth,
   elevation, range, and optional range-rate observations. The streaming sampler
   preserves the batch sensor's random sequence and cadence. Raw Gaussian
   draws are retained with a `valid` disposition and reason; impossible
   angular/range draws remain evidence but are excluded before packet delivery
   or observation conversion. Truth diagnostics are removed before valid
   observations reach the receiver.
3. Downlink samples the enabled spacecraft resource channels: temperature,
   battery SOC/energy, solar generation, demand, served/unmet load, and battery
   charge/discharge power. These are **ideal onboard sensor readings** in this
   version; there is no additional telemetry noise, quantization, or smoothing.
4. Packets are delivered after `latency_s`. In-flight packets may arrive after
   contact ends. There is no onboard recorder or store-and-forward backlog.
5. A nonlinear tracking EKF updates the ground orbit and covariance at the
   **measurement epoch**. Display predictions are separate from the posterior,
   so displaying a later time does not prevent a delayed update. Observations
   older than the latest posterior are rejected and audited; there is no
   out-of-sequence smoothing. Duplicate station/epoch observations are rejected.
6. Between observations the receiver predicts state and covariance with OEL's
   existing two-body RK4 estimator, bounded by `prediction_step_s`. Process noise
   entries are diagonal variances added per second. It has no access to truth
   forces, maneuvers, or faults. Health channels retain their last received values.

Cadences must be integer multiples of `simulator.dt_s`. The session uses this
fixed step when ground segment is enabled. Packet reception timestamps retain
physical arrival time; processing occurs at the first simulation sample at or
after that time. Snapshot cadence follows simulator samples, independent of the
tracking and downlink cadences. Times, ages, and outages use **simulation time**.
Outage intervals include the start and exclude the end.

## API and evidence

`SimulationSession.reset()`, `.step()`, and `SimulationResult.snapshot(index)`
expose `.ground_segment` with `time_s`, `contacts`, `objects`, and `packets`.
Each object has an optional `orbit`, a telemetry channel mapping, and last-received
`tracking` data by station (retained even when no prior is available). Orbit data
include prediction epoch, posterior epoch, state, covariance, age, status, frame,
units, prediction model, and a stale flag. Channels carry measurement time,
reception time, source station, age, stale flag, and received/held status. A stale
threshold is a display policy, not a confidence or accuracy guarantee.

The standalone `sim.ground_segment.GroundSegment` accepts validated delivered
packets through `advance(time_s, packets=..., contacts=...)`; it never accepts a
truth state. External adapters are responsible for contact/delivery gating.
`SimulatedGroundSegment` provides OEL's truth-facing adapter. Returned snapshots
are independent copies. Multiple stations share one posterior per object.

Completed runs write `ground_segment.json`. With review enabled, query:

```sql
SELECT time_s, station_id, object_id, tracking, downlink, uplink
FROM ground_segment_contacts ORDER BY time_s, station_id, object_id
```

`ground_segment_state` contains `orbit_json`, `telemetry_json`, and `tracking_json`;
`ground_segment_packets` records measurement/reception/processing times,
disposition and received data. Dynamic sessions prune snapshot evidence with the
retained simulation history while preserving the live posterior and held values.

## Scope

This first version supports Earth-centered satellite scenarios. Connectivity is
geometric access plus explicit service outages; **RF link-budget thresholds,
antenna acquisition, weather, bandwidth limits, packet loss, and command execution
are not modeled by this layer yet**. Uplink is an availability indicator only.
Truth may use higher fidelity forces, but ground prediction is two-body. Ground
estimation remains experimental, particularly near angle-coordinate singularities
and with weak geometry or poor priors. It is a reusable simulation capability,
not an operational flight-dynamics qualification.
