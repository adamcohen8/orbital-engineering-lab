"""Modern GP/OMM catalogs and representable TLEs, without Cartesian relabeling."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass

import numpy as np

from sim.dynamics.orbit.epoch import julian_date_to_datetime
from sim.dynamics.orbit.ogp import ogp_propagate_teme_batch_reference
from sim.dynamics.orbit.tle import TLEElements, ogp_mean_elements_from_mapping, parse_tle_lines
from sim.frame_time import epoch_julian_date

from .common import json_bytes, read_text, strict_json
from .contracts import MAX_OBJECTS, DataIOError, Trajectory
from .time import creation_time, parse_time

FIELDS = {
    "MEAN_MOTION": "mean_motion_rev_per_day",
    "ECCENTRICITY": "eccentricity",
    "INCLINATION": "inclination_deg",
    "RA_OF_ASC_NODE": "raan_deg",
    "ARG_OF_PERICENTER": "argp_deg",
    "MEAN_ANOMALY": "mean_anomaly_deg",
    "BSTAR": "bstar",
    "MEAN_MOTION_DOT": "mean_motion_derivative_rev_per_day2",
    "MEAN_MOTION_DDOT": "mean_motion_second_derivative_rev_per_day3",
    "ELEMENT_SET_NO": "element_number",
    "REV_AT_EPOCH": "revolution_number",
    "NORAD_CAT_ID": "norad_number",
    "CLASSIFICATION_TYPE": "classification",
    "EPHEMERIS_TYPE": "ephemeris_type",
}
REQUIRED = {
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
}


@dataclass(frozen=True)
class CatalogRecord:
    elements: TLEElements
    fields: dict

    @property
    def object_id(self):
        return str(
            self.fields.get("NORAD_CAT_ID") or self.fields.get("OBJECT_ID") or self.fields.get("OBJECT_NAME") or ""
        )


def from_gp_fields(fields: dict, *, provider_profile="celestrak-gp") -> CatalogRecord:
    """Only the named CelesTrak GP profile permits its documented omissions."""
    value = dict(fields)
    if any(isinstance(value.get(key), bool) for key in FIELDS):
        raise DataIOError("GP fields must not use booleans as numeric values.")
    missing = REQUIRED - value.keys()
    if missing:
        raise DataIOError(f"Missing GP fields: {sorted(missing)}.")
    defaults = {"CENTER_NAME": "EARTH", "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC", "MEAN_ELEMENT_THEORY": "SGP4"}
    for key, expected in defaults.items():
        if key not in value:
            if provider_profile != "celestrak-gp":
                raise DataIOError(f"Missing OMM field {key}; provider defaults require an explicit profile.")
            value[key] = expected
        if str(value[key]).upper() != expected:
            raise DataIOError(f"OGP requires {key}={expected}.")
    if str(value.get("EPHEMERIS_TYPE", "0")) not in {"", "0"}:
        raise DataIOError(
            "Only ordinary SGP4/SDP4 GP ephemeris type 0 is supported; XP/other theories require their own owner."
        )
    if "GM" in value and float(value["GM"]) != 398600.8:
        raise DataIOError(
            "OGP GP materialization requires its declared WGS-72 GM; another gravity convention is not silently substituted."
        )
    cat = str(value.get("NORAD_CAT_ID", ""))
    if cat and (not cat.isdigit() or not 1 <= int(cat) <= 999_999_999):
        raise DataIOError("NORAD_CAT_ID must be a positive integer of at most nine digits.")
    epoch = parse_time(str(value["EPOCH"]), "UTC")
    mapping = {dest: value[key] for key, dest in FIELDS.items() if key in value}
    mapping.update(
        epoch_jd_utc=epoch_julian_date(epoch, "UTC"), international_designator=str(value.get("OBJECT_ID", ""))
    )
    for key in ("RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"):
        if not math.isfinite(float(value[key])) or not 0 <= float(value[key]) < 360:
            raise DataIOError(f"{key} must be in [0,360).")
    if not any(value.get(key) for key in ("NORAD_CAT_ID", "OBJECT_ID", "OBJECT_NAME")):
        raise DataIOError("A catalog record requires an object identifier.")
    try:
        elements = ogp_mean_elements_from_mapping(mapping)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataIOError(f"Invalid GP mean elements: {exc}") from exc
    return CatalogRecord(elements, value)


def from_elements(elements: TLEElements, *, object_name=None, object_id=None) -> CatalogRecord:
    value = {key: getattr(elements, dest) for key, dest in FIELDS.items()}
    value.update(
        EPOCH=julian_date_to_datetime(elements.epoch_jd_utc).isoformat().replace("+00:00", "Z"),
        OBJECT_NAME=object_name or elements.norad_number or "OEL OBJECT",
        OBJECT_ID=object_id if object_id is not None else elements.international_designator,
    )
    return from_gp_fields(value)


def read_catalog(path, *, format: str, provider_profile="celestrak-gp") -> list[CatalogRecord]:
    return parse_catalog(read_text(path), format=format, provider_profile=provider_profile)


def parse_catalog(raw: str, *, format: str, provider_profile="celestrak-gp") -> list[CatalogRecord]:
    """Parse serialized catalog bytes without an intermediate filesystem artifact."""
    if format == "gp-json":
        values = strict_json(raw)
        values = values if isinstance(values, list) else [values]
    elif format == "gp-csv":
        reader = csv.DictReader(io.StringIO(raw))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise DataIOError("Catalog CSV requires unique column names.")
        values = list(reader)
    elif format in {"omm-kvn", "omm-xml"}:
        from .navigation import parse_navigation

        document = parse_navigation(raw, format=format)
        from sim.interchange.ccsds_odm import OmmMessage

        if not isinstance(document, OmmMessage):
            raise DataIOError("Expected an OMM.")
        values = [
            {
                **document.mean_elements,
                **document.tle_parameters,
                "OBJECT_ID": document.metadata.object_id,
                "OBJECT_NAME": document.metadata.object_name,
                "CENTER_NAME": document.metadata.center_name,
                "REF_FRAME": document.metadata.ref_frame,
                "TIME_SYSTEM": document.metadata.time_system,
                "MEAN_ELEMENT_THEORY": document.metadata.mean_element_theory,
                **document.user_defined,
            }
        ]
        for field in ("creation_date", "originator", "message_id", "classification"):
            if (item := getattr(document.header, field)) is not None:
                values[0][field.upper()] = item
        if document.covariance is not None:
            from sim.interchange.ccsds_odm import _COVARIANCE_KEYS, _COVARIANCE_POSITIONS

            values[0].update(
                {key: document.covariance.matrix[i][j] for key, (i, j) in zip(_COVARIANCE_KEYS, _COVARIANCE_POSITIONS)}
            )
            if document.covariance.ref_frame:
                values[0]["COV_REF_FRAME"] = document.covariance.ref_frame
    elif format == "tle":
        lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
        result = []
        index = 0
        while index < len(lines):
            name = None
            if not lines[index].startswith("1 "):
                name = lines[index].removeprefix("0 ")
                index += 1
            if index + 1 >= len(lines):
                raise DataIOError("Incomplete TLE pair.")
            elements = parse_tle_lines(lines[index], lines[index + 1], require_checksum=True)
            result.append(from_elements(elements, object_name=name))
            index += 2
        if not result or len(result) > MAX_OBJECTS:
            raise DataIOError("Empty or oversized TLE catalog.")
        return result
    else:
        raise DataIOError(f"Not a catalog format: {format}.")
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= MAX_OBJECTS
        or any(not isinstance(v, dict) for v in values)
    ):
        raise DataIOError("Expected a bounded catalog of mappings.")
    records = [from_gp_fields(value, provider_profile=provider_profile) for value in values]
    identities = [(r.object_id, r.elements.epoch_jd_utc) for r in records]
    if len(set(identities)) != len(identities):
        raise DataIOError("Duplicate object/epoch catalog records.")
    return records


def encode_tle(record: CatalogRecord) -> str:
    """Use the independent SGP4 package's standard formatter; never truncate IDs."""
    e = record.elements
    if not str(e.norad_number).isdigit() or not 1 <= int(e.norad_number) <= 99999:
        raise DataIOError("TLE requires a representable five-digit NORAD ID; export OMM/GP for larger IDs.")
    dt = julian_date_to_datetime(e.epoch_jd_utc)
    if not 1957 <= dt.year <= 2056:
        raise DataIOError("Epoch is outside the unambiguous TLE year window 1957..2056.")
    if not 0 <= e.element_number <= 9999 or not 0 <= e.revolution_number <= 99999:
        raise DataIOError("TLE element/revolution number cannot be represented.")
    try:
        from sgp4.api import WGS72, Satrec
        from sgp4.exporter import export_tle
    except ImportError as exc:
        raise DataIOError("TLE formatting requires the 'sgp4' IO dependency.") from exc
    sat = Satrec()
    # SGP4's init inputs use radians/minute and actual derivative conventions.
    sat.sgp4init(
        WGS72,
        "i",
        int(e.norad_number),
        e.epoch_jd_utc - 2433281.5,
        e.bstar,
        e.mean_motion_derivative_rev_per_day2 * 2 * math.pi / 1440**2,
        e.mean_motion_second_derivative_rev_per_day3 * 2 * math.pi / 1440**3,
        e.eccentricity,
        math.radians(e.argp_deg),
        math.radians(e.inclination_deg),
        math.radians(e.mean_anomaly_deg),
        e.mean_motion_rev_per_day * 2 * math.pi / 1440,
        math.radians(e.raan_deg),
    )
    sat.classification = e.classification or "U"
    designator = str(e.international_designator)
    if "-" in designator and len(designator.split("-")[0]) == 4:
        designator = designator[2:].replace("-", "")
    if len(designator) > 8 or not designator.isascii():
        raise DataIOError("International designator cannot be represented in a TLE.")
    sat.intldesg = designator
    sat.elnum, sat.revnum = e.element_number, e.revolution_number
    lines = export_tle(sat)
    parse_tle_lines(*lines, require_checksum=True)
    if any(len(line) != 69 for line in lines):
        raise DataIOError("TLE field width overflow.")
    return "\n".join(lines) + "\n"


def encode_catalog(records: list[CatalogRecord], *, format: str) -> bytes:
    if not records or len(records) > MAX_OBJECTS:
        raise DataIOError("A bounded nonempty catalog is required.")
    if format == "gp-json":
        return json_bytes([r.fields for r in records])
    if format == "gp-csv":
        keys = list(dict.fromkeys(k for r in records for k in r.fields))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(r.fields for r in records)
        return stream.getvalue().encode()
    if format == "tle":
        if any(r.fields.get("CLASSIFICATION") for r in records):
            raise DataIOError("TLE cannot preserve a message classification banner; export OMM/GP instead.")
        if any(
            any(key.startswith(("CX_", "CY_", "CZ_")) or key == "COV_REF_FRAME" for key in r.fields) for r in records
        ):
            raise DataIOError("TLE cannot preserve supplied covariance; export OMM/GP instead.")
        return "".join(encode_tle(record) for record in records).encode()
    if format in {"omm-kvn", "omm-xml"}:
        if len(records) != 1:
            raise DataIOError("The bounded OMM profile writes one object per file; use GP JSON/CSV for catalogs.")
        r = records[0]
        values = {"CCSDS_OMM_VERS": "3.0", "CREATION_DATE": creation_time(), "ORIGINATOR": "OEL", **r.fields}
        # SGP4-family products use WGS-72. This bounded OMM owner requires GM;
        # preserve the source value or state the family constant explicitly.
        values.setdefault("GM", 398600.8)
        element_keys = list(FIELDS)
        element_keys.insert(element_keys.index("BSTAR"), "GM")
        keys = [
            "CCSDS_OMM_VERS",
            "CREATION_DATE",
            "ORIGINATOR",
            "CLASSIFICATION",
            "MESSAGE_ID",
            "OBJECT_NAME",
            "OBJECT_ID",
            "CENTER_NAME",
            "REF_FRAME",
            "TIME_SYSTEM",
            "MEAN_ELEMENT_THEORY",
            "EPOCH",
            *element_keys,
        ]
        from sim.interchange.ccsds_odm import _COVARIANCE_KEYS

        keys += ["COV_REF_FRAME", *_COVARIANCE_KEYS]
        supported = set(keys)
        extras = set(values) - supported
        if any(not key.startswith("USER_DEFINED_") for key in extras):
            raise DataIOError(f"OMM output would lose provider-specific fields: {sorted(extras)}.")
        kvn = "\n".join(f"{key} = {values[key]}" for key in [*keys, *sorted(extras)] if key in values) + "\n"
        from sim.interchange.ccsds_odm import parse_odm_kvn

        from .navigation import encode_navigation

        return encode_navigation(parse_odm_kvn(kvn), format=format)
    raise DataIOError(f"Unsupported catalog output: {format}.")


def propagate_catalog(record: CatalogRecord, *, start_epoch: str, duration_s: float, step_s: float) -> Trajectory:
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(step_s) or step_s <= 0:
        raise DataIOError("duration_s and step_s must be finite and positive.")
    count = int(np.ceil(duration_s / step_s)) + 1
    if count > 200_000:
        raise DataIOError("Requested OGP history exceeds the sample limit.")
    epoch = parse_time(start_epoch)
    times = np.unique(np.minimum(np.arange(count) * step_s, duration_s))
    jd = epoch_julian_date(epoch, "UTC") + times / 86400
    result = ogp_propagate_teme_batch_reference([record.elements], (jd - record.elements.epoch_jd_utc) * 1440)
    if not np.all(result.success):
        raise DataIOError(f"OGP propagation failed: {result.errors.tolist()}.")
    positions, velocities = result.position_teme_km[0], result.velocity_teme_km_s[0]
    return Trajectory(
        record.object_id,
        "EARTH",
        "TEME",
        epoch,
        times,
        np.hstack((positions, velocities)),
        str(record.fields.get("OBJECT_NAME", record.object_id)),
        metadata={"gp_input": record.fields, "propagation": "OEL OGP"},
    )
