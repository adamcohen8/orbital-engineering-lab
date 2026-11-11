"""Delivered-temperature investigation and guarded, observation-verified procedures."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict
from uuid import uuid4

from .contracts import digest
from .gateway import CommandGate
from .temperature_contracts import (
    THERMAL_NUMERIC,
    THERMAL_PARAMETERS,
    ProcedureAction,
    TemperatureReading,
    ThermalProcedurePolicy,
    utc_seconds,
)
from .yamcs import TelemetryPublisher


class TemperatureHistory:
    def __init__(self, policy: ThermalProcedurePolicy, *, wall_time=time.time):
        self.policy, self.wall_time = policy, wall_time
        self.rows = deque(maxlen=128)
        self.last = None

    def ingest(self, observation, package_digest):
        v = observation["values"]
        if v["temperature_package_digest"] != package_digest:
            raise ValueError("temperature package mismatch")
        if type(v["temperature_sequence"]) not in (int, float) or v["temperature_sequence"] % 1:
            raise ValueError("invalid temperature sequence")
        row = TemperatureReading(
            v["temperature_component"],
            v["temperature_sensor"],
            v["temperature_source"],
            v["temperature_clock"],
            v["temperature_source_time_s"],
            int(v["temperature_sequence"]),
            v["temperature_measured_at"],
            v["component_temperature_c"],
            v["temperature_quality"],
        )
        if (row.component, row.sensor, row.source) != (self.policy.component, self.policy.sensor, self.policy.source):
            raise ValueError("temperature source is outside configured procedure scope")
        age = self.wall_time() - utc_seconds(row.measured_at)
        transport_age = self.wall_time() - utc_seconds(observation["generation_time"])
        if min(age, transport_age) < -0.5 or max(age, transport_age) > self.policy.max_telemetry_age_s:
            raise ValueError("stale or future temperature observation")
        if row.quality != "valid":
            self.rows.clear()
            self.last = row
            raise ValueError("temperature quality is invalid")
        if self.last and row.clock == self.last.clock:
            if row.sequence == self.last.sequence:
                if row != self.last:
                    self.rows.clear()
                    raise ValueError("conflicting duplicate temperature sample")
                return row  # Cached sample cannot count as another measurement.
            if row.sequence < self.last.sequence or row.source_time_s <= self.last.source_time_s:
                raise ValueError("out-of-order temperature sample")
            if row.source_time_s - self.last.source_time_s > self.policy.maximum_gap_s:
                self.rows.clear()
        elif self.last:
            self.rows.clear()  # A new clock requires a new evidence window.
        self.last = row
        self.rows.append(row)
        return row

    def trend(self):
        rows = list(self.rows)[-self.policy.minimum_samples :]
        if (
            len(rows) < self.policy.minimum_samples
            or rows[-1].source_time_s - rows[0].source_time_s < self.policy.minimum_span_s
        ):
            return {
                "status": "insufficient_evidence",
                "reason": "need distinct valid samples spanning the trend window",
                "samples": len(rows),
            }
        span = rows[-1].source_time_s - rows[0].source_time_s
        rate = (rows[-1].temperature_c - rows[0].temperature_c) / span
        return {
            "status": "available",
            "trend": "rising" if rate > 0 else ("cooling" if rate < 0 else "steady"),
            "rate_c_per_s": rate,
            "span_s": span,
            "samples": len(rows),
            "temperature_c": rows[-1].temperature_c,
            "method": "observed endpoint slope; no forecast",
        }


class _ProcedureEvaluator:
    def __init__(self, service):
        self.service = service

    def evaluate(self, action, observation):
        return self.service.assess(action, observation)


class TemperatureService:
    def __init__(self, package, reader, source, device, policy, evidence_dir, log):
        self.reader, self.source, self.device, self.policy, self.log = reader, source, device, policy, log
        self.session_id = str(uuid4())
        self.identity = digest(
            {"spacecraft": package.identity, "procedure_policy": policy.identity, "procedure_session": self.session_id}
        )
        self.history = TemperatureHistory(policy)
        self.publisher = TelemetryPublisher()
        self.packet, self.last_publication = 0, None
        self.gate = CommandGate(
            self, policy, self.observe, evidence_dir / "procedure-ledger.sqlite", evaluator=_ProcedureEvaluator(self)
        )
        self.receipts = {}
        self.command_ids = {}
        for request_id, result in self.gate.db.execute("SELECT id,result FROM requests"):
            self.receipts[request_id] = json.loads(result)

    def describe(self):
        return {
            "component": self.policy.component,
            "component_code": self.policy.component_code,
            "procedures": ["suspend_payload", "resume_payload"],
            "policy": asdict(self.policy),
            "procedure_package_digest": self.identity,
            "procedure_session_id": self.session_id,
            "temperature_prediction": "unavailable",
            "verification": "distinct delivered post-command observations; stable cool window before resume",
            "adapter": type(self.device).__name__,
            "source_adapter": type(self.source).__name__,
            "modeled_consequences": "this workflow does not model payload thermal or electrical effects",
        }

    def publish(self, mission_time_s):
        row = self.source.read(mission_time_s)
        if row is None:
            return
        mode = self.device.observed_mode()
        if mode not in ("operating", "standby"):
            raise ValueError("invalid observed payload mode")
        key = (row, mode)
        if key == self.last_publication:
            return  # Do not freshen cached measurements on every plant step.
        self.packet += 1
        values = {
            "component_temperature_c": row.temperature_c,
            "temperature_source_time_s": row.source_time_s,
            "temperature_sequence": row.sequence,
            "temperature_component": row.component,
            "temperature_sensor": row.sensor,
            "temperature_source": row.source,
            "temperature_clock": row.clock,
            "temperature_measured_at": row.measured_at,
            "temperature_quality": row.quality,
            "payload_mode": mode,
            "temperature_package_digest": self.identity,
            "temperature_packet_id": str(self.packet),
        }
        generated = self.publisher.publish(values, fields=THERMAL_PARAMETERS, numeric=THERMAL_NUMERIC)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                observation = self.reader.observe_temperature()
                if observation["values"]["temperature_packet_id"] == str(self.packet):
                    self.last_publication = key
                    try:
                        self.history.ingest(observation, self.identity)
                    except ValueError as exc:
                        self.log("temperature_rejected", {"reason": str(exc)})
                    self.log("temperature_delivered", observation)
                    return
            except (ValueError, OSError):
                pass
            self.publisher.publish(values, generated, fields=THERMAL_PARAMETERS, numeric=THERMAL_NUMERIC)
            time.sleep(0.03)
        raise TimeoutError("Yamcs did not deliver temperature observation")

    def observe(self):
        observation = self.reader.observe_temperature()
        self.history.ingest(observation, self.identity)
        return observation

    def investigate(self):
        try:
            observation = self.observe()
            row = self.history.last
            return {
                "disposition": "observed",
                "observation": observation,
                "trend": self.history.trend(),
                "condition": "hot" if row.temperature_c >= self.policy.suspend_at_c else "below_suspend_threshold",
                "temperature_prediction": "unavailable",
                "limits": "threshold/trend evidence does not diagnose the physical fault",
            }
        except (ValueError, OSError) as exc:
            return {"disposition": "insufficient_evidence", "reason": str(exc)}

    def assess(self, action, observation):
        if action.component_code != self.policy.component_code:
            return {"disposition": "rejected", "reason": "component outside procedure scope"}
        mode = observation["values"]["payload_mode"]
        if action.procedure == "suspend_payload":
            if mode != "operating" or self.history.last.temperature_c < self.policy.suspend_at_c:
                return {
                    "disposition": "rejected",
                    "reason": "suspend requires observed operating mode and hot component",
                }
        else:
            if mode != "standby" or not any(self.verify(i).get("status") == "recovered" for i in self.receipts):
                return {
                    "disposition": "insufficient_evidence",
                    "reason": "resume requires verified post-suspend cool window",
                }
        return {
            "disposition": "permitted",
            "reason": "configured procedure preconditions satisfied",
            "approval_reference": self.policy.approval_reference,
            "observation_digest": digest(observation),
            "policy_digest": self.policy.identity,
            "temperature_prediction": "unavailable",
        }

    def evaluate(self, action):
        try:
            result = self.gate.evaluate(action)
        except (ValueError, OSError) as exc:
            result = {"disposition": "insufficient_evidence", "reason": str(exc)}
        self.log("procedure_evaluation", {"action": action.to_dict(), "result": result})
        return result

    def receive(self, data, mission_time_s):
        action, ticket = ProcedureAction.decode(data)
        result = self.gate.admit(action, ticket)
        if result["disposition"] == "permitted":
            row = self.history.last
            mode = "standby" if action.procedure == "suspend_payload" else "operating"
            accepted = self.device.set_mode(mode, action.request_id)
            if accepted.get("status") != "accepted":
                raise RuntimeError("device did not acknowledge the procedure; intent remains indeterminate")
            self.publish(mission_time_s)
            observed = self.observe()
            result = {
                "disposition": "applied",
                "action": action.to_dict(),
                "device_acknowledgment": accepted,
                "expected_mode": mode,
                "baseline": asdict(row),
                "gate_evidence": result,
                "observed_mode": observed["values"]["payload_mode"],
            }
            self.gate.record(action, result)
        self.receipts[action.request_id] = result
        command_id = self.reader.find_command(action, ticket)
        if command_id:
            self.command_ids[action.request_id] = command_id
            self.reader.annotate(
                command_id,
                {
                    "Ack_OEL_Procedure_Gate_Status": "OK"
                    if result["disposition"] in ("applied", "duplicate")
                    else "NOK",
                    "Ack_OEL_Device_Status": "OK" if result["disposition"] == "applied" else "NA",
                },
                command_name="/oel/ThermalProcedure",
            )
        self.log("procedure_receipt", {"request_id": action.request_id, "result": result})

    def verify(self, request_id):
        receipt = self.receipts.get(request_id)
        if not receipt:
            return {"status": "pending"}
        # A duplicate transport receipt must not discard the original evidence.
        if receipt.get("disposition") == "duplicate":
            receipt = receipt["prior"]
        if receipt.get("disposition") != "applied":
            return {"status": "not_applied", "receipt": receipt}
        try:
            observation = self.observe()
        except (ValueError, OSError) as exc:
            return {"status": "insufficient_evidence", "reason": str(exc), "receipt": receipt}
        v, baseline = observation["values"], receipt["baseline"]
        if v["temperature_clock"] != baseline["clock"]:
            return {
                "status": "insufficient_evidence",
                "reason": "source clock changed; reconcile procedure",
                "receipt": receipt,
            }
        if v["payload_mode"] != receipt["expected_mode"]:
            return {"status": "mode_unconfirmed", "receipt": receipt, "observation": observation}
        rows = [r for r in self.history.rows if r.clock == baseline["clock"] and r.sequence > baseline["sequence"]]
        if receipt["action"]["procedure"] == "resume_payload":
            status = (
                "resumed_observed" if rows and rows[-1].temperature_c <= self.policy.resume_below_c else "monitoring"
            )
        else:
            cool = []
            for row in reversed(rows):
                if row.temperature_c > self.policy.resume_below_c:
                    break
                cool.append(row)
            cool.reverse()
            stable = (
                len(cool) >= self.policy.minimum_samples
                and cool[-1].source_time_s - cool[0].source_time_s >= self.policy.recovery_span_s
                and all(b.temperature_c <= a.temperature_c for a, b in zip(cool, cool[1:]))
            )
            elapsed = self.history.last.source_time_s - baseline["source_time_s"]
            status = (
                "recovered"
                if stable
                else ("failed_to_cool" if rows and elapsed >= self.policy.verification_timeout_s else "monitoring")
            )
        result = {
            "status": status,
            "receipt": receipt,
            "observation": observation,
            "post_command_samples": len(rows),
            "trend": self.history.trend(),
            "temperature_prediction": "unavailable",
        }
        command_id = self.command_ids.get(request_id)
        if command_id:
            self.reader.annotate(command_id, {"OEL_Temperature_Outcome": status}, command_name="/oel/ThermalProcedure")
        self.log("procedure_verification", {"request_id": request_id, "result": result})
        return result

    def close(self):
        self.publisher.close()
        self.gate.close()
