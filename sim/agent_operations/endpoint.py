"""OEL spacecraft endpoint, command lease, and local engineering tool service."""

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError

from sim.api import SimulationSession

from .contracts import PARAMETERS, PointingAction, Policy
from .engineering import SpacecraftPackage, command_event, onboard_fields, utc_epoch
from .gateway import CommandGate
from .temperature import TemperatureService
from .temperature_contracts import PROCEDURE_WIRE, ProcedureAction
from .yamcs import TelemetryPublisher, YamcsClient


class OperationsEndpoint:
    def __init__(
        self,
        package: SpacecraftPackage,
        policy: Policy,
        reader: YamcsClient,
        evidence_dir: Path,
        *,
        tm_port=11916,
        tc_port=11925,
        temperature_source=None,
        payload_device=None,
        temperature_policy=None,
    ):
        self.package, self.policy, self.reader = package, policy, reader
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = evidence_dir / "endpoint.jsonl"
        self.trace = self.trace_path.open("a", buffering=1)
        self.lock = threading.RLock()
        self.session = SimulationSession(package.config(), history_mode="dynamic", max_history_samples=16)
        self.session.reset()
        self.snap = self.session.step()
        self.status, self.last_request = "idle", ""
        self.active = None
        self.receipts = {}
        self.command_ids = {}
        self.sequence = 0
        self.sample = 0
        self.temperature = None
        if temperature_source is not None:
            if payload_device is None or temperature_policy is None:
                raise ValueError("temperature workflow requires a source, device adapter and procedure policy")
            self.temperature = TemperatureService(
                package, reader, temperature_source, payload_device, temperature_policy, evidence_dir, self.log
            )
        self.publisher = TelemetryPublisher(tm_port)
        self.gate = CommandGate(package, policy, reader.observe, evidence_dir / "command-ledger.sqlite")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", tc_port))
        self.socket.settimeout(0.5)
        self.stop = threading.Event()
        self.publish()
        self.thread = threading.Thread(target=self.receive, daemon=True)
        self.thread.start()

    def log(self, kind, value):
        self.trace.write(
            json.dumps(
                {
                    "schema": "oel.operations_trace.v1",
                    "wall_time": datetime.now(timezone.utc).isoformat(),
                    "kind": kind,
                    "value": value,
                },
                allow_nan=False,
            )
            + "\n"
        )

    def publish(self):
        fields = onboard_fields(self.session, self.package.object_id)
        values = {name: fields[name] for name in PARAMETERS if name in fields}
        for i, axis in enumerate("xyz"):
            values[f"wheel_{axis}_momentum"] = fields[f"wheel_{i}_momentum_n_m_s"]
        self.sample += 1
        values.update(
            {
                "sim_time_s": self.snap.time_s,
                "battery_soc": self.snap.spacecraft_resources[self.package.object_id]["battery_soc"],
                "spacecraft_id": self.package.spacecraft_id,
                "package_digest": self.package.identity,
                "mode": fields["adcs_operational_mode"],
                "command_status": self.status,
                "last_request_id": self.last_request,
                "sample_id": str(self.sample),
                "epoch_utc": utc_epoch(self.package.config()["simulator"]["initial_jd_utc"] + self.snap.time_s / 86400),
            }
        )
        generated = self.publisher.publish(values)
        self.log("telemetry_sent", values)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                observation = self.reader.observe()
                if observation["values"]["sample_id"] == str(self.sample):
                    if self.temperature:
                        self.temperature.publish(self.snap.time_s)
                    return observation
            except (ValueError, OSError, URLError):
                pass
            # UDP drops are retried with the original source generation time.
            self.publisher.publish(values, generated)
            time.sleep(0.03)
        raise TimeoutError("Yamcs did not deliver the coherent telemetry sample")

    def evaluate(self, action):
        with self.lock:
            result = self.gate.evaluate(action)
            self.log("evaluation", {"action": action.to_dict(), "result": result})
            return result

    def receive(self):
        while not self.stop.is_set():
            try:
                data, _ = self.socket.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                try:
                    if self.temperature and len(data) == PROCEDURE_WIRE.size:
                        self.temperature.receive(data, self.snap.time_s)
                        continue
                    action, ticket = PointingAction.decode(data)
                    result = self.gate.admit(action, ticket)
                    self.log("command_gate", {"action": action.to_dict(), "result": result})
                    if result["disposition"] == "permitted":
                        before = onboard_fields(self.session, self.package.object_id)["target_update_count"]
                        self.sequence += 1
                        self.session.publish_fsw_input(
                            self.package.object_id,
                            command_event(
                                action, self.snap.time_s, sequence=self.sequence, object_id=self.package.object_id
                            ),
                        )
                        self.snap = self.session.step(dt_s=0.1)
                        after = onboard_fields(self.session, self.package.object_id)["target_update_count"]
                        if after != before + 1:
                            raise RuntimeError("flight software did not accept exactly one target update")
                        self.active = {"action": action, "expires": self.snap.time_s + action.hold_s}
                        self.status, self.last_request = "active", str(action.request_id)
                        result = {
                            "disposition": "applied",
                            "request_id": action.request_id,
                            "fsw_target_updates": after,
                            "lease_expires_sim_s": self.active["expires"],
                            "gate_evidence": result,
                        }
                        self.active["receipt"] = result
                        self.gate.record(action, result)
                        self.publish()
                    self.receipts[action.request_id] = result
                    self.report_history(action, ticket, result)
                    self.log("command_receipt", result)
                except Exception as exc:
                    # Fail closed. An intent already logged remains indeterminate.
                    self.log("command_error", {"error_type": type(exc).__name__, "message": str(exc)})

    def advance(self, seconds):
        if type(seconds) not in (int, float) or not 0 < seconds <= 150:
            raise ValueError("simulation wait must be in (0, 150] seconds")
        with self.lock:
            goal = self.snap.time_s + seconds
            while self.snap.time_s < goal - 1e-8 and not self.session.done:
                boundary = min(goal, self.active["expires"]) if self.active else goal
                self.snap = self.session.step(dt_s=min(1.0, boundary - self.snap.time_s))
                fields = onboard_fields(self.session, self.package.object_id)
                battery = self.snap.spacecraft_resources[self.package.object_id]["battery_soc"]
                violation = (
                    battery < self.policy.battery_reserve
                    or fields["wheel_momentum_fraction"] > self.policy.wheel_fraction_limit
                    or fields["angular_rate_norm_rad_s"] > self.policy.max_rate_rad_s
                    or fields["adcs_operational_mode"] != "nominal"
                )
                if self.active and (violation or self.snap.time_s >= self.active["expires"] - 1e-8):
                    action = self.active["action"]
                    outcome = {
                        "request_id": action.request_id,
                        "hold_end_error_rad": fields["attitude_error_rad"],
                        "battery_soc": battery,
                        "status": "aborted_constraint"
                        if violation
                        else ("completed" if fields["attitude_error_rad"] <= 0.03 else "pointing_not_achieved"),
                    }
                    self.sequence += 1
                    recovery = PointingAction(action.request_id, self.package.recovery_target, 1.0)
                    self.session.publish_fsw_input(
                        self.package.object_id,
                        command_event(
                            recovery,
                            self.snap.time_s,
                            sequence=self.sequence,
                            command_id=f"recovery-{action.request_id}",
                            object_id=self.package.object_id,
                        ),
                    )
                    self.receipts[action.request_id] = {**self.active["receipt"], "outcome": outcome}
                    self.active = None
                    self.status = outcome["status"]
                    self.gate.record(action, self.receipts[action.request_id])
                    if action.request_id in self.command_ids:
                        self.reader.annotate(
                            self.command_ids[action.request_id],
                            {
                                "Ack_OEL_Hold_Status": "OK" if outcome["status"] == "completed" else "NOK",
                                "OEL_Outcome": outcome["status"],
                            },
                        )
                    self.log("command_outcome", outcome)
                self.publish()
            observation = self.reader.observe()
            if self.temperature:
                observation["temperature_workflow"] = self.temperature.investigate()
            return observation

    def report_history(self, action, ticket, result):
        try:
            command_id = self.reader.find_command(action, ticket)
            if command_id:
                self.command_ids.setdefault(action.request_id, command_id)
                self.reader.annotate(
                    command_id,
                    {
                        "Ack_OEL_Gate_Status": "OK" if result["disposition"] in ("applied", "duplicate") else "NOK",
                        "Ack_OEL_FlightSoftware_Status": "OK" if result["disposition"] == "applied" else "NA",
                        "OEL_Reason": result.get("reason", result["disposition"]),
                    },
                )
            else:
                self.log("history_not_yet_available", {"request_id": action.request_id})
        except URLError as exc:
            self.log("history_error", {"request_id": action.request_id, "error": str(exc)})

    def close(self):
        self.stop.set()
        self.socket.close()
        self.thread.join(timeout=2)
        self.publisher.close()
        self.gate.close()
        if self.temperature:
            self.temperature.close()
        self.trace.close()


def serve(endpoint: OperationsEndpoint, port=18091):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("invalid request size")
                body = json.loads(self.rfile.read(length))
                if self.path == "/evaluate":
                    result = endpoint.evaluate(PointingAction.from_dict(body))
                elif self.path == "/advance" and set(body) == {"seconds"}:
                    result = endpoint.advance(body["seconds"])
                elif self.path == "/evaluate-procedure" and endpoint.temperature:
                    with endpoint.lock:
                        result = endpoint.temperature.evaluate(ProcedureAction.from_dict(body))
                else:
                    raise ValueError("unknown tool request")
                self.reply(200, result)
            except Exception as exc:
                self.reply(400, {"disposition": "insufficient_evidence", "error": str(exc)})

        def do_GET(self):
            try:
                with endpoint.lock:
                    if self.path == "/capabilities":
                        from dataclasses import asdict

                        result = {
                            **endpoint.package.describe(),
                            "policy": asdict(endpoint.policy),
                            "policy_digest": endpoint.policy.identity,
                            "simulation_clock": "stepped by wait tool; wall-clock freshness enforced",
                            "temperature_workflow": endpoint.temperature.describe()
                            if endpoint.temperature
                            else {"status": "unavailable"},
                        }
                    elif self.path == "/temperature" and endpoint.temperature:
                        result = endpoint.temperature.investigate()
                    elif self.path.startswith("/procedure-receipt/") and endpoint.temperature:
                        result = endpoint.temperature.verify(int(self.path.rsplit("/", 1)[-1]))
                    elif self.path.startswith("/receipt/"):
                        result = endpoint.receipts.get(int(self.path.rsplit("/", 1)[-1]), {"disposition": "pending"})
                    else:
                        raise ValueError("unknown tool request")
                    self.reply(200, result)
            except Exception as exc:
                self.reply(400, {"error": str(exc)})

        def reply(self, status, result):
            payload = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
