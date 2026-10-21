"""External visualization products from explicitly framed saved trajectories."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from sim.utils.geodesy import ecef_to_geodetic_deg_km

from .common import json_bytes
from .contracts import DataIOError, Trajectory
from .time import at, format_time


def encode_visualization(t: Trajectory, *, format: str) -> tuple[bytes, dict]:
    if t.center.upper() != "EARTH":
        raise DataIOError("These visualization profiles require Earth-centered data.")
    omissions = ["Velocity and covariance remain in the source evidence; this is a position visualization."]
    if format == "czml":
        if t.frame not in {"GCRF", "ITRF"}:
            raise DataIOError("CZML requires explicit GCRF inertial or ITRF fixed states; transform EME2000 first.")
        values = np.column_stack((t.times_s, t.states[:, :3] * 1000)).reshape(-1).tolist()
        availability = format_time(at(t.epoch, t.times_s[0])) + "/" + format_time(at(t.epoch, t.times_s[-1]))
        document = [
            {"id": "document", "version": "1.0", "name": "OEL saved trajectory"},
            {
                "id": "oel:" + t.object_id,
                "name": t.object_name or t.object_id,
                "availability": availability,
                "position": {
                    "epoch": format_time(t.epoch),
                    "referenceFrame": "FIXED" if t.frame == "ITRF" else "INERTIAL",
                    "cartesian": values,
                    "interpolationAlgorithm": "LINEAR",
                    "interpolationDegree": 1,
                },
                "point": {"pixelSize": 5},
                "path": {"width": 1, "show": True},
            },
        ]
        return json_bytes(document), {"visualization_omissions": omissions, "interpolation": "linear position only"}
    if t.frame != "ITRF":
        raise DataIOError("GeoJSON/KML requires explicitly transformed ITRF positions.")
    positions = []
    for pv in t.states:
        if np.linalg.norm(pv[:3]) <= 1e-9:
            raise DataIOError("Earth-center position has no defined geodetic latitude/longitude.")
        latitude, longitude, altitude = ecef_to_geodetic_deg_km(pv[:3])
        positions.append([longitude, latitude, altitude * 1000])
    if format == "geojson":
        features = []
        for time, coordinates in zip(t.times_s, positions):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": coordinates},
                    "properties": {
                        "object_id": t.object_id,
                        "epoch_utc": format_time(at(t.epoch, time)),
                        "altitude_reference": "WGS84 ellipsoid, metres",
                    },
                }
            )
        return json_bytes({"type": "FeatureCollection", "features": features}), {"visualization_omissions": omissions}
    if format == "kml":
        root = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
        folder = ET.SubElement(root, "Document")
        for time, (lon, lat, _) in zip(t.times_s, positions):
            item = ET.SubElement(folder, "Placemark")
            ET.SubElement(item, "name").text = t.object_id
            ET.SubElement(ET.SubElement(item, "TimeStamp"), "when").text = format_time(at(t.epoch, time))
            point = ET.SubElement(item, "Point")
            ET.SubElement(point, "altitudeMode").text = "clampToGround"
            ET.SubElement(point, "coordinates").text = f"{lon:.12g},{lat:.12g}"
        ET.indent(root)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True), {
            "visualization_omissions": [
                *omissions,
                "KML represents ground-track points; spacecraft ellipsoidal altitude is not mislabeled as height above sea level.",
            ]
        }
    raise DataIOError(f"Unsupported visualization format: {format}.")
