# CCSDS OEM Interoperability

OEL provides a bounded public adapter for CCSDS Orbit Ephemeris Message (OEM)
3.0 files in Keyword Value Notation (KVN). The adapter is an interchange
surface around OEL evidence; it is not part of the deterministic physics
engine and never executes a scenario.

The supported profile identifier is `oel.ccsds-oem-kvn.v0.2`.

The separate [modular IO adapter](data-io.md) adds bounded OEM 2.0/3.0 KVN/XML
conversion and saved-review exports. The legacy profile below retains its
existing contract and import behavior.

## Supported Profile

The v0.2 reader and writer support:

- CCSDS OEM version `3.0` in UTF-8 KVN;
- one object per OEM, with one or more ephemeris segments for inspection and
  round-trip work;
- mandatory Cartesian position and velocity in kilometres and kilometres per
  second;
- optional acceleration components when every state row contains them;
- optional Cartesian 6x6 covariance blocks in lower-triangular KVN form,
  including matrix epoch and optional covariance reference frame;
- calendar and day-of-year absolute epochs through microsecond precision;
- metadata, header comments, data comments before state rows, and the standard
  interpolation fields; and
- bounded input size, line, segment, and state counts.

The narrower OEL import profile accepts exactly one continuous segment with:

- `CENTER_NAME = EARTH`;
- `REF_FRAME = EME2000`;
- `TIME_SYSTEM = UTC`; and
- no `REF_FRAME_EPOCH`.

An import-ready EME2000 state is mapped to the OEL `OEL/ECI/J2000`
`frames-v1` convention without a numerical rotation. Generic `ECI`, TEME,
GCRF, ITRF, or another frame is never silently relabeled as EME2000.
Use the explicit `convert-oem` command when an Earth-centered OEM state and
covariance must be converted into the supported EME2000/UTC profile.

## Commands

Inspect an OEM without changing it:

```bash
python -m sim.ccsds inspect-oem path/to/input.oem --json
```

Parse and deterministically reserialize a message:

```bash
python -m sim.ccsds roundtrip-oem path/to/input.oem \
  --output outputs/interchange/roundtrip.oem
```

Compare two messages using semantic metadata and frozen state tolerances:

```bash
python -m sim.ccsds compare-oem first.oem second.oem \
  --output outputs/interchange/oem-comparison.json --json
```

Create a mission-input packet from an import-ready OEM:

```bash
python -m sim.ccsds import-oem path/to/input.oem \
  --output outputs/agent_inputs/oem_packet.json
```

The resulting packet uses the first ephemeris sample as scenario initial
state, preserves the complete source span and mapping provenance, and warns
that it does not replay, interpolate, or fit the full ephemeris.

Explicitly convert an Earth-centered OEM before import:

```bash
python -m sim.ccsds convert-oem path/to/gcrf_input.oem \
  --target-frame EME2000 --target-time-system UTC \
  --eop path/to/finals2000A.all \
  --output outputs/interchange/eme2000_utc.oem --json
```

The command transforms every state and covariance, converts metadata/data
epochs, removes the source interpolation declaration, and writes a
content-bound receipt. ITRF or UT1 work requires an epoch-covering EOP source.
Acceleration-bearing messages fail closed when the frame changes because the
bounded converter does not yet implement second-order frame kinematics.

Export one object's completed-run review history:

```bash
python -m sim.ccsds export-oem outputs/my_completed_run \
  --object-id satellite \
  --object-name "MISSION SATELLITE" \
  --originator OEL \
  --output outputs/interchange/mission_satellite.oem --json
```

Export requires a stable review store, a verified source configuration with an
absolute `simulator.initial_jd_utc`, strictly increasing state samples, and
canonical ECI review-state evidence. When a matching
`object_state_covariance` row is present, export also requires the complete
ECI `[x,y,z,vx,vy,vz]` matrix, declared units, mathematical-validity status,
and calibration scope. It emits both the OEM and
`<name>.oem.receipt.json`. The receipt binds the source review-store hash,
object, state count, frame mapping, output hash, semantic read-back result, and
non-claims. Export does not propagate or execute anything.

## Validation

The retained public fixture manifest is
`sim/interchange/examples/ccsds_oem_validation_manifest.json`. It binds:

- a complete synthetic OEM 3.0 KVN positive fixture;
- an official CCSDS 502.0-B-3 Annex G covariance example used for positive
  parse, round-trip, and independent-consumer checks;
- parser, serializer, malformed-input, resource-boundary, semantic-comparison,
  import, and completed-run-export tests; and
- an offline Orekit 13.1.7 cross-read report, bound to the current OEM
  implementation, with pinned JAR and Orekit-data identities.

The external cross-reader checks message/segment/state counts, selected object
and frame/time metadata, the first and last fixtures' `X` and `VY` values, and
three selected covariance elements plus covariance frame/count. Other
supported fields are covered by OEL parse/serialize and semantic round-trip
tests, not by the retained Orekit cross-read.

Routine replay does not invoke an external tool:

```bash
python -m pytest -q sim/tests/test_ccsds_oem.py
```

Maintainers with the pinned open-source Orekit runtime may refresh the
independent-consumer result explicitly:

```bash
python -m tools.validate_ccsds_oem \
  --oem sim/interchange/examples/oel_earth_eme2000_utc_v3.oem \
  --output sim/interchange/examples/ccsds_oem_orekit_13_1_7_acceptance.json \
  --orekit-root path/to/pinned/orekit
```

No STK, ODTK, or paid service is required for this profile.

## Explicit Non-Claims

The v0.2 profile does not support:

- treating mathematical covariance validity as estimator calibration;
- OEM XML;
- non-Earth center conversion;
- online SANA identifier or originator validation;
- full ephemeris replay as a simulator truth source;
- interpolation-accuracy or orbit-accuracy claims; or
- OCM, TDM, CDM, RINEX, SP3, SPICE, STK, or GMAT formats.

Those are later interoperability increments and must receive their own bounded
contracts and validation evidence.

Bounded OPM/OMM support is documented separately in
[`docs/ccsds-odm.md`](ccsds-odm.md).
