"""Yamcs JSON parameter ingress and constrained HTTP client."""

from __future__ import annotations

import base64
import json
import socket
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .contracts import NUMERIC_FIELDS, PARAMETERS, PointingAction
from .temperature_contracts import THERMAL_NUMERIC, THERMAL_PARAMETERS, ProcedureAction


class YamcsClient:
    def __init__(self, url: str, username: str, password: str, *, instance="oel", processor="realtime"):
        parsed = urlparse(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username
            or parsed.query
            or parsed.path not in ("", "/")
        ):
            raise ValueError("this local exercise requires a loopback Yamcs URL")
        self.url, self.instance, self.processor = url.rstrip("/"), instance, processor
        self.authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        self.opener = build_opener(ProxyHandler({}))

    def request(self, method, path, body=None):
        request = Request(
            self.url + path,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": self.authorization, "Content-Type": "application/json"},
            method=method,
        )
        with self.opener.open(request, timeout=10) as response:
            content = response.read()
        return json.loads(content) if content else {}

    @property
    def path(self):
        return f"/api/processors/{quote(self.instance, safe='')}/{quote(self.processor, safe='')}"

    def observe(self):
        return self._observe(PARAMETERS, NUMERIC_FIELDS, "oel.delivered_operations_observation.v1")

    def observe_temperature(self):
        return self._observe(THERMAL_PARAMETERS, THERMAL_NUMERIC, "oel.delivered_temperature_observation.v1")

    def _observe(self, fields, numeric, schema):
        reply = self.request(
            "POST",
            self.path + "/parameters:batchGet",
            {"id": [{"name": "/oel/" + name} for name in fields], "fromCache": True},
        )
        values, times = {}, set()
        for record in reply.get("value", []):
            if record.get("acquisitionStatus", "ACQUIRED") != "ACQUIRED":
                raise ValueError("Yamcs telemetry is not acquired")
            name = record["id"]["name"].rsplit("/", 1)[-1]
            value = record.get("engValue", {})
            key = "doubleValue" if name in numeric else "stringValue"
            if key not in value or name in values:
                raise ValueError("invalid or duplicate Yamcs parameter")
            values[name] = value[key]
            times.add(record.get("generationTime"))
        if set(values) != set(fields) or len(times) != 1 or None in times:
            raise ValueError("missing or incoherent Yamcs telemetry snapshot")
        return {
            "schema": schema,
            "source": "yamcs",
            "generation_time": times.pop(),
            "values": values,
        }

    def issue(self, action: PointingAction, ticket: int):
        return self.request(
            "POST",
            self.path + "/commands/" + quote("/oel/PointTarget", safe=""),
            {
                "args": {
                    "request_id": action.request_id,
                    "ticket": ticket,
                    "target_x": action.target_eci_m[0],
                    "target_y": action.target_eci_m[1],
                    "target_z": action.target_eci_m[2],
                    "hold_s": action.hold_s,
                },
                "origin": "oel-operations-agent",
                "sequenceNumber": action.request_id,
            },
        )

    def issue_procedure(self, action: ProcedureAction, ticket: int):
        return self.request(
            "POST",
            self.path + "/commands/" + quote("/oel/ThermalProcedure", safe=""),
            {
                "args": {
                    "request_id": action.request_id,
                    "ticket": ticket,
                    "component_code": action.component_code,
                    "operation_code": action.operation_code,
                },
                "origin": "oel-operations-agent",
                "sequenceNumber": action.request_id,
            },
        )

    def annotate(self, command_id: str, attributes: dict, *, command_name="/oel/PointTarget"):
        return self.request(
            "POST",
            self.path + "/commandhistory/" + quote(command_name, safe=""),
            {
                "id": command_id,
                "attributes": [
                    {"name": k, "value": {"type": "STRING", "stringValue": str(v)}} for k, v in attributes.items()
                ],
            },
        )

    def find_command(self, action: PointingAction, ticket: int):
        if isinstance(action, ProcedureAction):
            command_name = "oel/ThermalProcedure"
            expected = {
                "request_id": action.request_id,
                "ticket": ticket,
                "component_code": action.component_code,
                "operation_code": action.operation_code,
            }
        else:
            command_name = "oel/PointTarget"
            expected = {
                "request_id": action.request_id,
                "ticket": ticket,
                "target_x": action.target_eci_m[0],
                "target_y": action.target_eci_m[1],
                "target_z": action.target_eci_m[2],
                "hold_s": action.hold_s,
            }
        reply = self.request("GET", f"/api/archive/{quote(self.instance, safe='')}/commands?limit=50")
        for record in reply.get("commands", []):
            args = {
                item["name"]: item["value"].get("uint32Value", item["value"].get("doubleValue"))
                for item in record.get("assignments", [])
            }
            if record["commandName"].lstrip("/") == command_name and args == expected:
                return record["id"]
        return None


class TelemetryPublisher:
    def __init__(self, port=11916):
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, values: dict, generation_time: str | None = None, *, fields=PARAMETERS, numeric=NUMERIC_FIELDS):
        generated = generation_time or datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )
        packet = {
            "parameter": [
                {
                    "id": {"name": "/oel/" + name},
                    "generationTime": generated,
                    "engValue": {"type": "DOUBLE", "doubleValue": values[name]}
                    if name in numeric
                    else {"type": "STRING", "stringValue": str(values[name])},
                }
                for name in fields
            ]
        }
        self.socket.sendto(json.dumps(packet, allow_nan=False).encode(), ("127.0.0.1", self.port))
        return generated

    def close(self):
        self.socket.close()
