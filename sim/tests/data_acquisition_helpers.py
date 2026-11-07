"""Synthetic acquisition inputs; no real accounts or provider responses."""
import csv
import io
from datetime import date, timedelta
from types import SimpleNamespace

from sim.data_acquisition.contracts import canonical
from sim.data_acquisition.service import AcquisitionService
from sim.data_acquisition.transport import Response
from sim.dynamics.orbit.space_weather_data import FIELDS

GRAVITY = b"""product_type gravity_field
modelname SYNTHETIC
earth_gravity_constant 3.986004415e14
radius 6378136.3
max_degree 2
norm fully_normalized
tide_system tide_free
end_of_head
gfc 0 0 1 0
gfc 2 0 -4.84165143790815e-4 0
"""


def gp(oid=123456789, day=10, **overrides):
    return {"NORAD_CAT_ID": oid, "OBJECT_NAME": "SYNTHETIC", "OBJECT_ID": "2024-001A", "CENTER_NAME": "EARTH",
        "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC", "MEAN_ELEMENT_THEORY": "SGP4", "EPOCH": f"2024-01-{day:02d}T00:00:00",
        "CREATION_DATE": f"2024-01-{day:02d}T01:00:00", "GP_ID": day, "MEAN_MOTION": 15.1, "ECCENTRICITY": 0.001,
        "INCLINATION": 45.0, "RA_OF_ASC_NODE": 5.0, "ARG_OF_PERICENTER": 10.0, "MEAN_ANOMALY": 20.0, "BSTAR": 0.0, **overrides}


def eop():
    header = "DATE,MJD,X,Y,UT1-UTC,LOD,DPSI,DEPS,DX,DY,DAT,DATA_TYPE\n"
    return (header + "".join(f"2024-01-{day},603{9 + day},0.1,0.2,0.03,0.001,0.04,0.05,0.006,0.007,37,{'P' if day == 14 else 'O'}\n" for day in range(10, 15))).encode()


def weather(*, missing=None, predicted=False):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for offset in range(7):
        day = date(2024, 1, 7) + timedelta(days=offset)
        row = dict.fromkeys(FIELDS, 1)
        row.update(DATE=day.isoformat(), **{"F10.7_DATA_TYPE": "PRD" if predicted else "OBS", "F10.7_ADJ_CENTER81": 150 + offset, "F10.7_OBS": 140 + offset})
        for i in range(1, 9):
            row[f"AP{i}"] = i + offset
        row["AP_AVG"] = 4 + offset
        if missing and day == missing[0]:
            row[missing[1]] = ""
        writer.writerow(row)
    return output.getvalue().encode()


class Secrets:
    def __init__(self):
        self.values = {}

    def set(self, reference, value):
        self.values[reference] = value

    def get(self, reference):
        return self.values[reference]

    def delete(self, reference):
        self.values.pop(reference)


class FakeNetwork:
    def __init__(self, payloads=()):
        self.payloads = list(payloads)
        self.calls = []
        self.authenticates = True

    def __call__(self, hosts, reserve=None, on_backoff=None, clock=None):
        network = self

        class Connection:
            jar = []

            def request(self, url, *, max_bytes, fields=None, product="request", retrieval_interval=0):
                if reserve:
                    reserve(product, retrieval_interval)
                network.calls.append((url, product))
                if fields is not None:
                    self.jar = [SimpleNamespace(name="synthetic-session", domain="www.space-track.org", is_expired=lambda: False)] if network.authenticates else []
                    return Response(b"{}", url, "application/json")
                value = network.payloads.pop(0)
                if isinstance(value, Exception):
                    raise value
                if isinstance(value, Response):
                    return value
                return Response(value if isinstance(value, bytes) else canonical(value), url, "application/json")

            def clear(self):
                self.jar = []

        return Connection()


def service(tmp_path, network=None, **kwargs):
    return AcquisitionService(data_root=tmp_path / "store", config_root=tmp_path / "connections", secrets=Secrets(),
        transport_factory=network or FakeNetwork(), **kwargs)


def history(**overrides):
    return {"provider": "spacetrack", "product": "gp_history", "connection": "personal", "object_ids": [123456789],
            "start": "2024-01-10T00:00:00Z", "stop": "2024-01-12T00:00:00Z", **overrides}
