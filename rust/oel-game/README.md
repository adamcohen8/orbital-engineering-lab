# OEL Rust game numerical core

`oel_rust_game` ABI 1 wheel for the default Trainer backend. Physics
continues through the separate `oel_rust_orbit` backend. See
`../../docs/rust-game-backend.md` for selection, build, and tested scope.

The FSW kernels receive onboard state/measurements and typed command values
packed by `sim.flight_software.rust_game_*`. Separate display and scoring
kernels receive game truth through their maintained game owners. Version 0.5.0 batches maintained boundary-record validation and internal
primitive conversion; external wrappers retain the Python validator.
Python owns event validation, mission/filter policy, actuator availability,
sequence/validity identifiers, telemetry, and persisted state. No opaque native
state needs serialization; immutable prediction schedules and transfer matrices
are recomputed from configuration after restore.

Pilot forces, geometry, PD/retreat feedback, prediction states and effector
positions use SI (metres, seconds, kilograms, newtons, radians). Quaternion
operations use scalar-first passive inertial-to-body quaternions, matching OEL.
The transfer feedback API follows the existing RIC transfer component's km,
km/s and km/s² units; the outer translation controller owns SI saturation.
Its subordinate acceleration limit may be positive infinity. Prediction rows
contain a row-major 6x6 transition matrix followed by a row-major 6x3 input
matrix from the maintained HCW model. Times are cumulative seconds, with the
pulse/horizon boundary schedule constructed by the shared model adapter.

The preview kernels retain the existing dashboard RK4 formulas and step grids,
using km, km/s, seconds and km³/s². Scoring uses km, km/s and km/s². The sampled
goal points arrive as little-endian float64 XYZ bytes; the original Python
curve generator and decision logic remain authoritative. Native internal conversion produces the same primitives for the maintained
Python JSON writer; public serialization keeps its Python owner.

The public repository includes the source and build manifests for this crate.
See [building public native source](../../docs/native-source-build.md) for
wheel build commands and host-specific qualification limits.
