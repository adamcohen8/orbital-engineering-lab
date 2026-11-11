# cFS SIL Starter Kit

This folder contains the OEL GNC v2 UDP transport, a native-Linux cFS 7.0.1
Software Bus adapter, and deterministic regression fixtures. The Python
transport implements `SatelliteFlightSoftware`. The maintained C reference
application accepts one canonical, valid `ideal_own_state.v1` measurement per
input batch and returns a canonical typed actuator command; it deliberately
rejects other event kinds and payloads rather than silently projecting them.

The normative wire contract is [`ICD_v2.md`](ICD_v2.md). `icd.yaml`,
`python_bridge.py`, `mock_cfs_endpoint.py`, and `sim_loop_adapter.py` preserve
the pre-v2 codec only as isolated compatibility-test fixtures; no maintained
satellite scenario or public integration entry point uses them.

## Data flow

1. A satellite selects `CfsV2UdpFlightSoftwareAdapter` as its complete stack.
2. OEL sends a CRC-protected v2 boot or input-batch packet containing canonical
   `sim.flight_software` JSON in SI units.
3. `oel_sil_bridge` validates the canonical reference-ADCS subset and v2 envelope, publishes an ordinary sensor
   message on the cFS Software Bus, and preserves the request identity.
4. `oel_adcs_app` receives that Software Bus message and publishes an actuator
   message.
5. The bridge returns a canonical `FlightSoftwareOutput` with a typed
   `ideal_wrench.v1` device command for the same invocation.
6. OEL validates the command on the ordinary actuator bus and the physical
   runtime realizes it.

cFS headers and compiler-aligned Software Bus structures are never treated as
wire bytes. Transport failures, malformed packets, and timeouts are bridge
failures; they are not modeled onboard or actuator faults.

## Mock bring-up

Run the maintained v2 mock endpoint and simulator together:

```bash
python integrations/cfs_sil/run_mock_sil_stack.py \
  --config configs/cfs_sil_single_sat.yaml
```

The mock endpoint implements the same boot/input/output semantics as the live
cFS mission and uses the same canonical Python codec as the in-process
adapter. The C codec is checked against byte-identical golden vectors.

## Install the cFS 7.0.1 apps

Validate a pinned checkout:

```bash
python tools/install_cfs_sil_bridge.py --cfs-root /path/to/cFS --check
```

Install the bridge and reference ADCS application:

```bash
python tools/install_cfs_sil_bridge.py \
  --cfs-root /path/to/cFS \
  --startup-script sample_defs/cpu1_cfe_es_startup.scr
```

The installer refuses unverified versions by default and updates mission files
idempotently. The maintained bridge is loopback-only unless its mission-owned
platform configuration is deliberately changed.

## Run the live reference ADCS mission

On a supported Linux host:

```bash
python tools/run_cfs_sil_mission.py --cfs-root /path/to/cFS
```

The launcher installs both apps, builds and starts `core-cpu1`, runs
`configs/cfs_sil_live_adcs.yaml`, and verifies:

- v2 boot/input/output packet identity and CRC checks;
- live cFS Software Bus sensor and command flow;
- typed, nonzero body-torque device commands;
- reduced attitude error with bounded body rate; and
- byte-identical packet evidence across repeated deterministic runs.

The deterministic invocation grid includes the initial and terminal boundaries:
N integration intervals produce N+1 input/output exchanges. Verification requires
contiguous invocation identities and one command timestamp matching each saved
boundary, so missing or duplicate exchanges cannot pass by count alone.

Results are saved in `cfs_live_mission_result.json` and
`cfs_reproducibility_manifest.json`. The full typed input, output, receipt, and
realization history remains in `master_run_summary.json` and `review/run.sqlite`.
Use `--skip-build` only after the same installed cFS tree has been built.

## Reference application

`cfs_adcs_app/` subscribes to `OEL_SIL_SENSOR_MID`, computes
`TorqueBodyNm = -kp * q_error_vec - kd * omega`, limits the torque norm, and
publishes `OEL_SIL_ACTUATOR_MID`. The UDP bridge owns transport only; the
control law still executes as a separate cFS Software Bus application.

This is an engineering SIL reference, not flight-qualified networking or
flight software. Mission use requires requirements-driven timing, fault,
security, hardware, and closed-loop validation for the exact deployment.
