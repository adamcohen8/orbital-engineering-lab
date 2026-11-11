# OEL cFS SIL Bridge App

`oel_sil_bridge` is the supported translation boundary between OEL's GNC v2
flight-software packets and cFS Software Bus messages. It targets the native Linux
configuration of cFS 7.0.1.

It does not cast UDP bytes to cFS structures. Incoming OEL packets are decoded
field-by-field, placed in a normal cFS telemetry message, and published on the
Software Bus. The reverse path subscribes to a normal cFS actuator-command
message, serializes its user data into the OEL wire format, and sends it to the
configured OEL endpoint.

## Default routing

| Direction | UDP packet | Software Bus message |
|---|---|---:|
| OEL to cFS | v2 boot/input batch | `OEL_SIL_SENSOR_MID` (`0x0900`) |
| cFS to OEL | v2 acknowledgement/output | `OEL_SIL_ACTUATOR_MID` (`0x1901`) |

The defaults are in `config/oel_sil_bridge_msgids.h`. Network settings and
queue/task settings are in `config/oel_sil_bridge_platform_cfg.h`.

## Add to a cFS 7.0.1 mission

Copy or link this directory into the cFS bundle as `apps/oel_sil_bridge`, then:

1. Add `oel_sil_bridge` to the target's `APPLIST` or
   `MISSION_GLOBAL_APPLIST` in `sample_defs/targets.cmake`.
2. Add this startup entry after the cFE services are available:

   ```text
   CFE_APP, oel_sil_bridge, OEL_SIL_BRIDGE_Main, OEL_SIL_BRIDGE, 50, 65536, 0x0, 0;
   ```

3. Build the native target using the cFS 7.0.1 build commands.
4. Start OEL with its bridge bound to `127.0.0.1:50100` and targeting the cFS
bridge at `127.0.0.1:50101`. See `../ICD_v2.md` for the wire envelope and
canonical payload contract.

The 64 KiB task stack is required because the bounded receive/decode path keeps
the maximum UDP datagram and its null-terminated JSON decode buffer in nested
stack frames. Reducing it below that measured call shape can terminate the
native cFS process with a stack fault during bridge startup.

The app intentionally defaults to loopback. Non-loopback operation requires a
mission-reviewed configuration change and the corresponding OEL opt-in.

`tools/install_cfs_sil_bridge.py` performs the app-list and generated-startup
updates automatically for the pinned cFS bundle. `--startup-script` additionally
updates a mission-maintained static startup script when one is used.

## Flight application contract

Flight applications subscribe to `OEL_SIL_SENSOR_MID` and publish
`OEL_SIL_ACTUATOR_MID`. They include `oel_sil_bridge_msg.h`; they do not depend
on the UDP codec or POSIX socket implementation.

The bridge is an engineering SIL interface, not flight-qualified networking
software. The current supported target is native Linux only.
