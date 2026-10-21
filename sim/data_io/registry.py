"""Static format contracts; availability and semantic use are separate fields."""

from __future__ import annotations

from importlib.util import find_spec


def formats():
    rows = []

    def add(ids, kind, profiles, *, read=True, write=True, options=(), dependency=None, limits=()):
        for identifier in ids:
            rows.append(
                {
                    "format": identifier,
                    "kind": kind,
                    "read": read,
                    "write": write,
                    "profiles": profiles,
                    "required_options": list(options),
                    "optional_dependency": dependency,
                    "dependency_available": dependency is None or find_spec(dependency) is not None,
                    "limits": list(limits),
                }
            )

    add(
        ("trajectory-json", "trajectory-csv", "trajectory-parquet"),
        "trajectory",
        "oel.trajectory.v1",
        limits=("CSV requires an explicit mapping/context or a hash-bound export receipt",),
    )
    next(r for r in rows if r["format"] == "trajectory-parquet").update(
        optional_dependency="pyarrow", dependency_available=find_spec("pyarrow") is not None
    )
    add(
        ("oem-kvn", "oem-xml"),
        "navigation",
        "CCSDS OEM 2.0/3.0 bounded Cartesian segments",
        limits=(
            "select segments explicitly",
            "trajectory conversion rejects acceleration loss",
            "no implied ephemeris replay",
        ),
    )
    add(
        ("opm-kvn", "opm-xml", "omm-kvn", "omm-xml"),
        "navigation",
        "CCSDS OPM/OMM 2.0/3.0 bounded profiles",
        limits=(
            "one object per file",
            "maneuver serialization does not execute maneuvers",
            "OMM can initialize OGP through materialization",
        ),
    )
    add(
        ("gp-json", "gp-csv", "tle"),
        "catalog",
        "CelesTrak GP / checksum-validated TLE",
        limits=("GP provider omissions require the named profile", "TLE output rejects unrepresentable IDs/epochs"),
    )
    next(r for r in rows if r["format"] == "tle").update(
        write_optional_dependency="sgp4", write_dependency_available=find_spec("sgp4") is not None
    )
    add(
        ("tdm-kvn", "tdm-xml"),
        "tracking",
        "CCSDS TDM 2.0 native observable preservation",
        limits=("AZEL/RADEC, range, Doppler, frequency and phase", "raw observables are not silently reduced for OD"),
    )
    add(
        ("cdm-kvn", "cdm-xml"),
        "navigation",
        "CCSDS CDM 1.0 two-object 6x6 RTN covariance",
        limits=("no extended drag/SRP/thrust covariance", "no operational disposition"),
    )
    add(
        ("aem-kvn", "apm-kvn", "stk-attitude"),
        "attitude",
        "ADM 2.0 quaternion/ANGVEL or STK quaternion/rate",
        limits=("explicit reference-to-body direction", "APM requires a single selected sample"),
    )
    add(
        ("stk-ephemeris",),
        "trajectory",
        "STK EphemerisTimePosVel",
        options=("object_id (read)",),
        limits=("named frame and central body", "no covariance syntax in this bounded profile"),
    )
    add(
        ("sp3",),
        "trajectory",
        "SP3-c/d PV",
        options=("frame (read)", "satellite_id (write)"),
        limits=(
            "explicit velocity derivation if absent",
            "regular samples for output",
            "clock values retained; not generated",
        ),
    )
    add(("sentinel-eof",), "trajectory", "Sentinel OSV PV", write=False, options=("frame", "object_id"))
    add(
        ("rinex-observation", "rinex-navigation"),
        "rinex",
        "RINEX 3.02..3.05",
        limits=(
            "navigation: eight-line GPS/Galileo records",
            "native observation LLI/SSI/clock retained",
            "no receiver OD or radiometric reduction",
        ),
    )
    add(
        ("spk",),
        "trajectory",
        "NAIF SPK type-13 writer / NAIF spacecraft reader",
        dependency="spiceypy",
        options=("kernels", "target_id", "center_id"),
        limits=("explicit frame and sample epochs for input", "no covariance"),
    )
    add(
        ("ck",),
        "attitude",
        "NAIF CK type-3 writer / rate-bearing CK reader",
        dependency="spiceypy",
        options=("kernels", "instrument_id", "clock_id"),
        limits=("explicit spacecraft-clock kernel", "reference-to-body orientation"),
    )
    add(
        ("czml", "geojson", "kml"),
        "visualization",
        "sampled positions",
        read=False,
        limits=("Earth frames only", "KML exports ground-track points", "visualization omits dynamics/covariance"),
    )
    add(("oel-state-product",), "state_product", "validated OEL Cartesian state envelope", write=False)
    add(
        ("normalized-json",),
        "structured",
        "oel.native-tracking.v1 / oel.rinex.v1",
        limits=("Native measurement/ephemeris values, not an OD solution",),
    )
    add(
        ("maneuver-plan-json",),
        "plan",
        "oel.maneuver-plan.v1",
        limits=("Normalized OPM integrated delta-v; no execution authorization or acceleration-profile inference",),
    )
    return rows


def capability_catalog():
    return {
        "schema": "oel.data-io-capabilities.v1",
        "formats": formats(),
        "operations": {
            "inspect": "read only",
            "convert": "read/write; no simulation or estimation",
            "export-review": "read completed review evidence and write a new artifact",
            "export-table": "bounded read-only SELECT and table serialization",
            "materialize": "write and validate a new scenario; no execution",
            "object": {
                "commands": ["describe-import", "import", "describe-export", "export"],
                "python_interface": "sim.data_io.objects",
                "builder_interface": "ScenarioBuilder.import_object",
                "continuation_owners": ["state-product", "satellite-checkpoint", "snapshot", "power-history"],
                "execution": False,
            },
            "fit-review": {
                "available": find_spec("sim.data_io.fit") is not None,
                "edition": "Pro",
                "entitlement": "tracking_od",
                "source": "completed review store",
                "simulation_executed": False,
            },
        },
    }
