"""Offline independent-consumer checks for the bounded data IO profiles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from sim.data_io.attitude import encode_adm
from sim.data_io.catalog import encode_catalog, read_catalog
from sim.data_io.common import source_receipt
from sim.data_io.contracts import AttitudeHistory, Trajectory
from sim.data_io.navigation import encode_navigation, oem_from_trajectory
from sim.data_io.precise import encode_sp3
from sim.data_io.products import opm_from_trajectory, tdm_from_measurements
from sim.data_io.time import parse_time

ROOT = Path(__file__).resolve().parents[1]


def cases():
    epoch = parse_time("2024-01-01T00:00:00")
    t = Trajectory(
        "2024-001A",
        "EARTH",
        "EME2000",
        epoch,
        [0, 30, 60],
        [[7000, 0, 0, 0, 7.5, 0], [7000, 225, 0, 0, 7.5, 0], [7000, 450, 0, 0, 7.5, 0]],
    )
    angle = np.array([0.0, 0.2, 0.4])
    h = AttitudeHistory(
        t.object_id,
        "EME2000",
        "SC_BODY_1",
        epoch,
        t.times_s,
        np.column_stack((np.cos(angle / 2), np.zeros((3, 2)), np.sin(angle / 2))),
        np.tile([0, 0, 0.2 / 30], (3, 1)),
    )
    products = []
    for kind, model in (("oem", oem_from_trajectory(t)), ("opm", opm_from_trajectory(t, sample_index=0))):
        for syntax in ("kvn", "xml"):
            products.append(
                (
                    f"{kind}-{syntax}",
                    kind,
                    encode_navigation(model, format=f"{kind}-{syntax}"),
                    3 if kind == "oem" else 1,
                )
            )
    record = read_catalog(ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm", format="omm-kvn")
    for syntax in ("kvn", "xml"):
        products.append((f"omm-{syntax}", "omm", encode_catalog(record, format=f"omm-{syntax}"), 1))
    products.append(("aem", "aem", encode_adm(h), 3))
    products.append(
        (
            "apm",
            "apm",
            encode_adm(replace(h, times_s=[0], quaternions=[[1, 0, 0, 0]], rates_rad_s=[[0, 0, 0]]), kind="apm"),
            1,
        )
    )
    rows = [
        {
            "epoch_utc": "2024-01-01T00:00:00",
            "observable": key,
            "value": value,
            "measurement_semantics": "reduced_geometric",
        }
        for key, value in (("ANGLE_1", 10), ("ANGLE_2", 20), ("RANGE", 100))
    ]
    message = tdm_from_measurements(rows, station_id="DSS-14", object_id="SAT")
    for syntax in ("kvn", "xml"):
        products.append((f"tdm-{syntax}", "tdm", encode_navigation(message, format=f"tdm-{syntax}"), 3))
    products.append(("sp3", "sp3", encode_sp3(replace(t, frame="ITRF"), satellite_id="G01"), 1))
    from sim.interchange.ccsds_cdm import CdmHeader, CdmMessage, CdmObject, CdmRelativeMetadata

    objects = []
    for i in (1, 2):
        metadata = {
            "OBJECT_DESIGNATOR": str(i),
            "CATALOG_NAME": "OEL",
            "OBJECT_NAME": f"SYNTHETIC-{i}",
            "INTERNATIONAL_DESIGNATOR": f"2024-00{i}A",
            "EPHEMERIS_NAME": "NONE",
            "COVARIANCE_METHOD": "CALCULATED",
            "MANEUVERABLE": "N/A",
            "ORBIT_CENTER": "EARTH",
            "REF_FRAME": "EME2000",
        }
        objects.append(
            CdmObject(
                f"OBJECT{i}", metadata, (7000.0, 0.0, float(i), 0.0, 7.5, 0.0), tuple(map(tuple, np.eye(6))), {}, {}, {}
            )
        )
    cdm = CdmMessage(
        CdmHeader("1.0", "2024-01-01T00:00:00", "OEL"),
        CdmRelativeMetadata("2024-01-01T00:00:00", 1000.0, 1.0, (0.0, 0.0, 1000.0), (0.0, 0.0, 1.0)),
        tuple(objects),
    )
    for syntax in ("kvn", "xml"):
        products.append((f"cdm-{syntax}", "cdm", encode_navigation(cdm, format=f"cdm-{syntax}"), 2))
    from sim.data_io.rinex import encode_rinex, parse_rinex

    def header(value, label):
        return f"{value:<60}{label}"

    obs = (
        "\n".join(
            [
                header("     3.05           O                   G", "RINEX VERSION / TYPE"),
                header("OEL                 OEL                 20240101 000000 UTC", "PGM / RUN BY / DATE"),
                header("SYNTHETIC", "MARKER NAME"),
                header("GEODETIC", "MARKER TYPE"),
                header("SYNTHETIC           OEL", "OBSERVER / AGENCY"),
                header("0000                SYNTHETIC           1.0", "REC # / TYPE / VERS"),
                header("0000                SYNTHETIC", "ANT # / TYPE"),
                header(f"{6378137.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "APPROX POSITION XYZ"),
                header(f"{0.0:14.4f}{0.0:14.4f}{0.0:14.4f}", "ANTENNA: DELTA H/E/N"),
                header("G    2 C1C L1C", "SYS / # / OBS TYPES"),
                header(f"{2024:6d}{1:6d}{1:6d}{0:6d}{0:6d}{0.0:13.7f}     GPS", "TIME OF FIRST OBS"),
                header("G L1C  0.00000  0", "SYS / PHASE SHIFT"),
                header("", "END OF HEADER"),
                "> 2024 01 01 00 00  0.0000000  0  1",
                "G01" + f"{20000000.0:14.3f}15{123456.0:14.3f} 7",
            ]
        )
        + "\n"
    )
    products.append(("rinex-observation", "rinex-observation", encode_rinex(parse_rinex(obs)), 1))

    def field(value):
        return f"{value:19.12E}".replace("E", "D")

    nav = (
        "\n".join(
            [
                header("     3.05           N                   G", "RINEX VERSION / TYPE"),
                header("OEL                 OEL                 20240101 000000 UTC", "PGM / RUN BY / DATE"),
                header("", "END OF HEADER"),
                "G01 2024 01 01 00 00 00" + "".join(field(v) for v in (1e-4, 1e-12, 0.0)),
                *["    " + "".join(field(v) for v in (0.0, 0.0, 0.0, 0.0)) for _ in range(7)],
            ]
        )
        + "\n"
    )
    products.append(("rinex-navigation", "rinex-navigation", encode_rinex(parse_rinex(nav)), 1))
    return products


def validate(orekit_root, output_dir):
    root = Path(orekit_root).resolve()
    jars = sorted((root / "lib").glob("*.jar"))
    data = root / "data/orekit-data-main"
    if not jars or not data.is_dir():
        raise ValueError("An existing local Orekit runtime with lib/ and data/orekit-data-main/ is required.")
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "tools/external_reference/OrekitDataIOAcceptance.java"
    classpath = os.pathsep.join(str(jar) for jar in jars)
    results = []
    with tempfile.TemporaryDirectory(prefix="oel-io-orekit-") as directory:
        subprocess.run(
            [shutil.which("javac") or "javac", "-cp", classpath, "-d", directory, str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        for name, family, content, expected in cases():
            path = output / name
            path.write_bytes(content)
            process = subprocess.run(
                [
                    shutil.which("java") or "java",
                    "-cp",
                    os.pathsep.join((directory, classpath)),
                    "OrekitDataIOAcceptance",
                    str(data),
                    family,
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            values = dict(line.split("=", 1) for line in process.stdout.splitlines() if "=" in line)
            checks = {
                "independent_parse": process.returncode == 0,
                "record_count": values.get("parsed_records") == str(expected),
            }
            if family in {"oem", "opm"}:
                checks["position_units"] = abs(float(values.get("first_x_m", "nan")) - 7_000_000) < 1e-6
            if family == "omm":
                checks["catalog_id"] = values.get("norad_id") == "99999"
            if family == "aem":
                checks["orientation_direction"] = (
                    abs(float(values.get("last_reference_x_body_y", "nan")) + np.sin(0.4)) < 1e-12
                )
                checks["body_rate_units"] = abs(float(values.get("last_body_rate_z", "nan")) - 0.2 / 30) < 1e-12
            results.append(
                {
                    "case": name,
                    "input": source_receipt(path),
                    "checks": {key: bool(value) for key, value in checks.items()},
                    "independent_values": values,
                    "error": process.stderr if process.returncode else None,
                }
            )
    report = {
        "schema": "oel.data-io-independent-consumer.v1",
        "status": "passed" if all(all(row["checks"].values()) for row in results) else "failed",
        "consumer_jars": [source_receipt(p) for p in jars],
        "validator_source": source_receipt(source),
        "adapter_sources": [source_receipt(p) for p in sorted((ROOT / "sim/data_io").glob("*.py"))],
        "data_root": str(data),
        "cases": results,
        "scope": "Independent parsing, record counts, and selected unit/identifier checks only; not full-field, orbit, attitude, or estimator accuracy qualification.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orekit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = validate(args.orekit_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": [{"case": r["case"], "checks": r["checks"], "error": r["error"]} for r in report["cases"]],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
