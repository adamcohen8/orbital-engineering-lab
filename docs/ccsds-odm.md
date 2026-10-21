# CCSDS OPM And OMM Interoperability

OEL provides a bounded public CCSDS 502.0-B-3 Orbit Parameter Message (OPM)
and Orbit Mean-Elements Message (OMM) 3.0 KVN profile. The contract identifier
is `oel.ccsds-opm-omm-kvn.v0.1`. These are inspection and interchange surfaces;
they do not execute a scenario.

The separate [modular IO adapter](data-io.md) adds bounded ODM 2.0/3.0 XML/KVN
conversion, GP catalogs, native OGP materialization and saved-result product
exports. Existing `sim.ccsds` contracts remain unchanged.

Supported OPM content includes one Cartesian state, optional Keplerian and
spacecraft parameters, one complete Cartesian covariance, repeated maneuver
blocks, comments, units, and `USER_DEFINED_*` fields. An Earth/EME2000/UTC OPM
can produce a mission-input packet from its state. Covariance and maneuver data
remain provenance: covariance is not claimed calibrated and maneuvers are not
scheduled or executed.

Supported OMM content includes mean elements, SGP4-family TLE parameters,
optional Cartesian covariance, units, comments, and user-defined fields. OMM
remains a mean-element catalog product; the adapter never silently converts it
to an osculating Cartesian state.

```bash
python -m sim.ccsds inspect-odm path/to/input.opm --json
python -m sim.ccsds roundtrip-odm path/to/input.omm \
  --output outputs/interchange/roundtrip.omm --json
python -m sim.ccsds compare-odm first.opm second.opm --json
python -m sim.ccsds import-opm path/to/input.opm \
  --output outputs/agent_inputs/opm_packet.json --json
```

The v0.1 profile supports UTF-8 KVN version 3.0 only. XML, multiple OPM/OMM
segments, OPM automatic maneuver execution, OMM propagation/materialization,
and complete CCSDS extension coverage remain outside the contract.

Routine validation is offline:

```bash
python -m pytest -q sim/tests/test_ccsds_odm.py
```

`sim/interchange/examples/ccsds_odm_validation_manifest.json` binds the OPM
and OMM fixtures, current ODM implementation, validator, and retained pinned
Orekit 13.1.7 cross-read evidence. The external OPM checks cover version,
object ID, frame, maneuver count, Cartesian position/velocity, and covariance
presence. The external OMM checks cover version, object ID, frame, mean-element
theory, mean motion, eccentricity, inclination, and NORAD catalog ID. OEL-only
tests cover the remaining supported fields and full semantic round trips;
those are not externally cross-read. Maintainers can refresh the independent
comparison without STK, ODTK, or another paid service:

```bash
python -m tools.validate_ccsds_odm \
  --output sim/interchange/examples/ccsds_odm_orekit_13_1_7_acceptance.json \
  --orekit-root path/to/pinned/orekit
```

Parser agreement does not establish orbit, maneuver, or covariance accuracy,
nor does it claim complete CCSDS 502.0-B-3 implementation.
