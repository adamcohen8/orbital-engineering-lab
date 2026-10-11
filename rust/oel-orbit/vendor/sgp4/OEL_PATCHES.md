# Pinned upstream and local changes

This is the MIT licensed sgp4 2.4.0 crate from the local Cargo registry.
The original per-file SHA-256 values are retained in UPSTREAM_SHA256.json;
LICENSE and the upstream Cargo manifests are preserved.

The OEL patch reuses paired sine/cosine evaluations in propagation. It keeps
all equations, constants, Kepler iteration limits, convergence tolerances,
error handling and AFSPC/improved mode contracts unchanged. The final
converged Kepler iteration's paired values are reused; if all ten iterations
update the angle, a fresh final pair is evaluated. Arithmetic terms
retain their original grouping.

The additive PreparedPropagation context hoists only the invariant near-Earth
inclination trig pair. The original Constants fields and serde schema remain
byte-for-byte unchanged in src/propagator.rs. Deep-space inclination is still
evaluated per sample. Both original and prepared calls share one authoritative
propagation body; no alternate equations or convergence policy are added.
