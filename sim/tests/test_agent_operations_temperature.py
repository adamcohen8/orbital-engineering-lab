"""Procedure and evidence checks; real Yamcs checks are opt-in."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sim.agent_operations.contracts import Policy
from sim.agent_operations.endpoint import OperationsEndpoint
from sim.agent_operations.engineering import SpacecraftPackage
from sim.agent_operations.temperature import TemperatureHistory, TemperatureService
from sim.agent_operations.temperature_contracts import ProcedureAction, TemperatureReading, ThermalProcedurePolicy
from sim.agent_operations.temperature_source import (
    PlaybackTemperatureSource,
    ReceivedTemperatureSource,
    SyntheticPayloadDevice,
)
from sim.agent_operations.yamcs import YamcsClient

FIXTURES = Path("sim/agent_operations/fixtures")


def observed(identity, *, sequence=1, seconds=0, temperature=80, mode="operating", quality="valid", clock="test"):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generation_time": now,
        "values": {
            "temperature_component": "payload",
            "temperature_sensor": "payload-thermistor-1",
            "temperature_source": "synthetic-playback-v1",
            "temperature_clock": clock,
            "temperature_source_time_s": seconds,
            "temperature_sequence": sequence,
            "temperature_measured_at": now,
            "component_temperature_c": temperature,
            "temperature_quality": quality,
            "payload_mode": mode,
            "temperature_package_digest": identity,
            "temperature_packet_id": str(sequence),
        },
    }


@pytest.fixture
def service(tmp_path):
    current = [None]
    reader = SimpleNamespace(observe_temperature=lambda: deepcopy(current[0]), find_command=lambda *_: None)
    device = SyntheticPayloadDevice()
    svc = TemperatureService(
        SimpleNamespace(identity="spacecraft"),
        reader,
        ReceivedTemperatureSource(),
        device,
        ThermalProcedurePolicy(),
        tmp_path,
        lambda *_: None,
    )
    current[0] = observed(svc.identity)

    # Fake delivery only for unit checks; opt-in checks exercise the real transport.
    def publish(_):
        current[0]["values"]["payload_mode"] = device.observed_mode()

    svc.publish = publish
    yield svc, current, device
    svc.close()


def apply_suspend(service):
    svc, _, _ = service
    action = ProcedureAction(101, 1, "suspend_payload")
    evaluation = svc.evaluate(action)
    assert evaluation["disposition"] == "permitted"
    svc.receive(action.encode(evaluation["ticket"]), 1)
    return action, evaluation


def deliver(service, sequence, seconds, temperature, **kwargs):
    svc, current, _ = service
    current[0] = observed(
        svc.identity,
        sequence=sequence,
        seconds=seconds,
        temperature=temperature,
        mode=kwargs.pop("mode", "standby"),
        **kwargs,
    )
    return svc.investigate()


def test_wire_and_observation_validation():
    action = ProcedureAction(1, 2, "suspend_payload")
    assert ProcedureAction.decode(action.encode(3)) == (action, 3)
    for invalid in (True, 0, -1, 2**32):
        with pytest.raises(ValueError):
            ProcedureAction(invalid, 1, "suspend_payload")
    with pytest.raises(ValueError):
        ProcedureAction(1, 1, "disable_heater")
    with pytest.raises(ValueError):
        ProcedureAction.from_dict(
            {"request_id": 1, "component_code": 1, "procedure": "suspend_payload", "override": True}
        )
    with pytest.raises(ValueError):
        ThermalProcedurePolicy(resume_below_c=80)
    with pytest.raises(ValueError):
        TemperatureReading("payload", "sensor", "source", "clock", 0, 1, "2026-10-02T00:00:00Z", float("nan"))


def test_cached_measurements_cannot_create_a_trend():
    history = TemperatureHistory(ThermalProcedurePolicy())
    sample = observed("package")
    for _ in range(10):
        history.ingest(sample, "package")
    assert history.trend()["status"] == "insufficient_evidence"
    history.ingest(observed("package", sequence=2, seconds=5, temperature=82), "package")
    history.ingest(observed("package", sequence=3, seconds=10, temperature=84), "package")
    assert history.trend()["trend"] == "rising"
    assert history.trend()["rate_c_per_s"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature_measured_at", "2000-01-01T00:00:00Z"),
        ("temperature_measured_at", "2099-01-01T00:00:00Z"),
        ("temperature_quality", "invalid"),
        ("temperature_sensor", "another-sensor"),
        ("temperature_component", "battery"),
        ("temperature_package_digest", "different"),
    ],
)
def test_untrusted_or_stale_measurement_cannot_authorize(service, field, value):
    svc, current, device = service
    current[0]["values"][field] = value
    result = svc.evaluate(ProcedureAction(1, 1, "suspend_payload"))
    assert result["disposition"] == "insufficient_evidence" and "ticket" not in result
    assert device.command_count == 0


def test_suspend_acknowledgment_is_not_recovery_and_resume_needs_stable_window(service):
    svc, current, device = service
    action, approval = apply_suspend(service)
    assert device.mode == "standby" and device.command_count == 1
    assert svc.verify(action.request_id)["status"] == "monitoring"
    resume = ProcedureAction(102, 1, "resume_payload")
    assert svc.evaluate(resume)["disposition"] == "insufficient_evidence"
    for _ in range(10):
        assert svc.verify(action.request_id)["status"] == "monitoring"
    deliver(service, 2, 10, 64)
    assert svc.evaluate(resume)["disposition"] == "insufficient_evidence"
    deliver(service, 3, 20, 63)
    deliver(service, 4, 30, 62)
    assert svc.verify(action.request_id)["status"] == "recovered"
    evaluation = svc.evaluate(resume)
    assert evaluation["disposition"] == "permitted"
    svc.receive(resume.encode(evaluation["ticket"]), 30)
    assert svc.verify(resume.request_id)["status"] == "monitoring"
    deliver(service, 5, 40, 62, mode="operating")
    assert svc.verify(resume.request_id)["status"] == "resumed_observed"
    # Replay through the same wire path is at most once, even with a consumed ticket.
    svc.receive(resume.encode(evaluation["ticket"]), 40)
    assert device.command_count == 2


def test_sample_change_invalidates_ticket_and_wrong_component_is_denied(service):
    svc, current, device = service
    action = ProcedureAction(1, 1, "suspend_payload")
    approval = svc.evaluate(action)
    current[0] = observed(svc.identity, sequence=2, seconds=5)
    svc.receive(action.encode(approval["ticket"]), 5)
    assert svc.receipts[1]["disposition"] == "rejected"
    assert device.command_count == 0
    assert svc.evaluate(ProcedureAction(2, 9, "suspend_payload"))["disposition"] == "rejected"


@pytest.mark.parametrize("disruption", ["invalid", "gap", "clock", "rising", "mode"])
def test_disrupted_recovery_window_never_authorizes_resume(service, disruption):
    svc, _, _ = service
    action, _ = apply_suspend(service)
    deliver(service, 2, 10, 64)
    deliver(service, 3, 20, 63)
    if disruption == "invalid":
        deliver(service, 4, 30, 62, quality="invalid")
    elif disruption == "gap":
        deliver(service, 4, 50, 62)
    elif disruption == "clock":
        deliver(service, 4, 30, 62, clock="reset")
    elif disruption == "rising":
        deliver(service, 4, 30, 64)
    else:
        deliver(service, 4, 30, 62, mode="operating")
    assert svc.verify(action.request_id)["status"] != "recovered"
    assert svc.evaluate(ProcedureAction(2, 1, "resume_payload"))["disposition"] != "permitted"


def test_failure_to_cool_and_fresh_transport_cannot_hide_old_measurement(service):
    svc, current, _ = service
    action, _ = apply_suspend(service)
    for index, seconds in enumerate(range(10, 101, 10), 2):
        deliver(service, index, seconds, 80 + index)
    assert svc.verify(action.request_id)["status"] == "failed_to_cool"
    current[0]["values"]["temperature_measured_at"] = (datetime.now(timezone.utc) - timedelta(seconds=100)).isoformat()
    current[0]["generation_time"] = datetime.now(timezone.utc).isoformat()
    assert svc.verify(action.request_id)["status"] == "insufficient_evidence"


def test_restart_after_durable_intent_does_not_repeat_device_command(service, tmp_path):
    svc, current, device = service
    action = ProcedureAction(3, 1, "suspend_payload")
    approval = svc.evaluate(action)
    assert svc.gate.admit(action, approval["ticket"])["disposition"] == "permitted"
    reader = SimpleNamespace(observe_temperature=lambda: deepcopy(current[0]), find_command=lambda *_: None)
    restarted = TemperatureService(
        SimpleNamespace(identity="spacecraft"),
        reader,
        ReceivedTemperatureSource(),
        device,
        svc.policy,
        tmp_path,
        lambda *_: None,
    )
    try:
        restarted.receive(action.encode(approval["ticket"]), 1)
        assert device.command_count == 0
        assert restarted.verify(3)["status"] == "not_applied"
    finally:
        restarted.close()


def test_new_session_cannot_use_previous_sessions_cached_temperature(service, tmp_path):
    svc, current, device = service
    reader = SimpleNamespace(observe_temperature=lambda: deepcopy(current[0]), find_command=lambda *_: None)
    restarted = TemperatureService(
        SimpleNamespace(identity="spacecraft"),
        reader,
        ReceivedTemperatureSource(),
        device,
        svc.policy,
        tmp_path / "new-session",
        lambda *_: None,
    )
    try:
        assert restarted.identity != svc.identity
        result = restarted.evaluate(ProcedureAction(4, 1, "suspend_payload"))
        assert result["disposition"] == "insufficient_evidence" and "package mismatch" in result["reason"]
        assert device.command_count == 0
    finally:
        restarted.close()


def test_external_source_preserves_measurement_and_playback_has_no_command_input():
    reading = TemperatureReading("payload", "sensor", "external", "sensor-clock", 20, 4, "2026-10-02T00:00:00Z", 80)
    source = ReceivedTemperatureSource()
    source.ingest(reading)
    assert source.read(100) is reading
    playback = PlaybackTemperatureSource(FIXTURES / "temperature_cooling.json", ThermalProcedurePolicy())
    a = playback.read(1)
    assert playback.read(4) is a
    assert playback.read(60).temperature_c == 62  # Values are fixture data, not cooling predictions.
    assert playback.read(100).quality == "invalid"  # Accelerated mission time cannot freshen a held sample.


@pytest.mark.external
@pytest.mark.skipif(os.environ.get("OEL_YAMCS_INTEGRATION") != "1", reason="requires local Yamcs")
@pytest.mark.parametrize(
    "fixture,expected", [("temperature_cooling.json", "recovered"), ("temperature_not_cooling.json", "failed_to_cool")]
)
def test_actual_yamcs_temperature_procedure(fixture, expected, tmp_path):
    credentials = json.loads(Path("build/agent-operations/yamcs/credentials.json").read_text())
    reader = YamcsClient("http://127.0.0.1:18090", "endpoint", credentials["endpoint"])
    agent = YamcsClient("http://127.0.0.1:18090", "agent", credentials["agent"])
    package = SpacecraftPackage(yaml.safe_load(Path("configs/agent_operations_pointing.yaml").read_text()))
    policy, device = ThermalProcedurePolicy(), SyntheticPayloadDevice()
    endpoint = OperationsEndpoint(
        package,
        Policy(),
        reader,
        tmp_path,
        temperature_source=PlaybackTemperatureSource(FIXTURES / fixture, policy),
        payload_device=device,
        temperature_policy=policy,
    )
    service = endpoint.temperature

    def wait_receipt(request_id):
        deadline = time.monotonic() + 10
        while request_id not in service.receipts and time.monotonic() < deadline:
            time.sleep(0.02)
        return service.receipts[request_id]

    try:
        endpoint.advance(10)
        assert service.investigate()["trend"]["trend"] == "rising"
        action = ProcedureAction(9101, 1, "suspend_payload")
        evaluation = service.evaluate(action)
        issued = agent.issue_procedure(action, evaluation["ticket"])
        assert wait_receipt(9101)["disposition"] == "applied"
        assert agent.observe_temperature()["values"]["payload_mode"] == "standby"
        assert service.verify(9101)["status"] == "monitoring"
        endpoint.advance(49 if expected == "recovered" else 90)
        assert service.verify(9101)["status"] == expected
        resume = ProcedureAction(9102, 1, "resume_payload")
        assert (service.evaluate(resume)["disposition"] == "permitted") == (expected == "recovered")
        agent.issue_procedure(action, evaluation["ticket"])
        deadline = time.monotonic() + 5
        while service.receipts[9101]["disposition"] != "duplicate" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service.receipts[9101]["disposition"] == "duplicate"
        assert device.command_count == 1
        history = agent.request("GET", "/api/archive/oel/commands/" + issued["id"])
        attrs = {a["name"]: a["value"]["stringValue"] for a in history["attr"] if a["name"].startswith("OEL_")}
        assert attrs["OEL_Temperature_Outcome"] == expected
        if expected == "recovered":
            approval = service.evaluate(resume)
            agent.issue_procedure(resume, approval["ticket"])
            assert wait_receipt(9102)["disposition"] == "applied"
            endpoint.advance(10)
            assert service.verify(9102)["status"] == "resumed_observed"
            assert device.command_count == 2
    finally:
        endpoint.close()
