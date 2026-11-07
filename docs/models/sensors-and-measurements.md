# Sensor And Measurement Model Reference

OEL's public sensor models create deterministic or seeded synthetic
measurements for simulation, controller bring-up, and evidence workflows. They
do not model calibrated hardware, association, communications transport,
operational tracking authority, or mission-qualified detection probability.

## Core Models And Units

| Model | Output vector | Units and frame |
| --- | --- | --- |
| `OwnStateSensor` / `NoisyOwnStateSensor` | `[x,y,z,vx,vy,vz]` | ECI km and km/s |
| `RelativeSensor(mode="angle_only")` | `[azimuth,elevation]` | radians from observer-to-target ECI LOS |
| `RelativeSensor(mode="range")` | `[range]` | km |
| `RelativeSensor(mode="range_rate")` | `[range,range_rate]` | km and km/s |
| `JointStateSensor` | position, velocity, quaternion, body rate | ECI km, ECI km/s, body-from-inertial unit quaternion, body rad/s |

`SensorNoiseConfig` applies seeded Gaussian `sigma`, additive `bias`,
`dropout_prob`, and acquisition-to-delivery `latency_s` where the model supports
latency. A first entry is broadcast when its vector does not match the
measurement dimension. Latent own-state measurements retain their acquisition
timestamp when released.

## Cadence And Access

`AccessConfig` can gate updates by positive `update_cadence_s`, positive
`max_range_km`, `fov_half_angle_rad` in `[0, pi]`, or `solid_angle_sr` in
`[0, 4*pi]`. These FOV forms are alternatives; configuring both is rejected
before access evaluation. Ground visibility requires a `GroundSite` plus a
`FrameContext` with an absolute epoch. The default FOV boresight is
observer radial-out; a supplied ECI boresight is normalized. Access failure
returns no measurement.

Knowledge tracking also supports a body-frame sensor mount offset and
boresight. The runtime transforms these using spacecraft attitude before its
synthetic range, LOS/FOV, and dropout gates. See
`docs/agent-observations.md` for the scenario-facing fields.

## Composition And Event Timing

`CompositeSensorModel` returns no measurement unless every child returns a
measurement and all acquisition timestamps agree within `time_tolerance_s`; it
otherwise concatenates child vectors in list order. `JointStateSensor` and
`NoisyOwnStateSensor` enforce their own update cadence before sampling.

`PhysicalSensorEventSource` packetizes an already-created
`MeasurementEvent` into the typed GNC-v2 `InputEvent` boundary. It requires the
sensor ID and sample/source clock to match, records a separate delivery clock,
and increments packet sequence even when the physical sample is absent. This
layer does not synthesize physics or expose truth to flight software.

## Evidence Boundary

Focused sensor tests cover dimensions, seeded noise/dropout/latency, access,
composition, body offset, and packet timing. Those tests establish software
contract behavior only. A real study must separately justify calibration,
bias stability, detection statistics, mounting knowledge, timing accuracy,
association, environment effects, and the estimator's applicability.
