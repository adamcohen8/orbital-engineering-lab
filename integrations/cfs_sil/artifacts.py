from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sim.utils.io import write_json


def write_bridge_artifacts(
    *,
    outdir: Path,
    bridge_hist: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str], dict[str, Any]]:
    rows = _packet_rows(bridge_hist)
    if not rows:
        return {}, {}

    summary = _summary(rows=rows, bridge_hist=bridge_hist)
    packet_log = outdir / "cfs_bridge_packets.jsonl"
    summary_json = outdir / "cfs_bridge_summary.json"
    summary_md = outdir / "cfs_bridge_summary.md"
    manifest_json = outdir / "cfs_reproducibility_manifest.json"
    packet_log.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    write_json(str(summary_json), summary)
    summary_md.write_text(_format_summary_md(summary), encoding="utf-8")
    manifest = _reproducibility_manifest(rows=rows, summary=summary)
    write_json(str(manifest_json), manifest)
    return (
        {
            "packet_log_jsonl": str(packet_log),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "reproducibility_manifest_json": str(manifest_json),
        },
        summary,
    )


def _reproducibility_manifest(*, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    canonical_rows = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    objects = dict(summary.get("objects", {}) or {})
    execution_modes = sorted({str(row.get("execution_mode", "unknown")) for row in objects.values()})
    timeout_policies = sorted({str(row.get("timeout_policy", "unknown")) for row in objects.values()})
    disqualifying_counts = {
        key: sum(int(row.get(key, 0)) for row in objects.values())
        for key in (
            "decode_error_count",
            "sequence_gap_count",
            "out_of_order_count",
            "timestamp_mismatch_count",
            "step_timeout_count",
        )
    }
    deterministic_step_qualified = bool(
        objects
        and execution_modes == ["deterministic_step"]
        and timeout_policies == ["error"]
        and not any(disqualifying_counts.values())
    )
    return {
        "schema": "oel_cfs_sil_reproducibility_manifest_v1",
        "wire_protocol_version": 1,
        "packet_evidence_sha256": hashlib.sha256(canonical_rows.encode("utf-8")).hexdigest(),
        "packet_count": len(rows),
        "execution_modes": execution_modes,
        "timeout_policies": timeout_policies,
        "command_validity_clock": "simulation_time",
        "disqualifying_counts": disqualifying_counts,
        "deterministic_step_qualified": deterministic_step_qualified,
    }


def _packet_rows(bridge_hist: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_id, events in sorted(dict(bridge_hist or {}).items()):
        for event in list(events or []):
            if not isinstance(event, dict):
                continue
            bridge = event.get("bridge", {})
            if not isinstance(bridge, dict) or bridge.get("source") != "cfs_sil_bridge":
                continue
            t_s = float(event.get("t_s", float("nan")))
            sensor = dict(bridge.get("last_sensor_state", {}) or {})
            if sensor.get("packet_hex"):
                rows.append(
                    {
                        "direction": "sim_to_cfs",
                        "object_id": str(object_id),
                        "t_s": t_s,
                        "seq": int(sensor.get("seq", -1)),
                        "sim_time_ns": int(sensor.get("sim_time_ns", 0)),
                        "packet_hex": str(sensor.get("packet_hex", "")),
                        "decoded": {k: v for k, v in sensor.items() if k not in {"packet_hex"}},
                    }
                )
            command = dict(bridge.get("last_actuator_command", {}) or {})
            if command.get("packet_hex") and bool(command.get("new_packet", True)):
                rows.append(
                    {
                        "direction": "cfs_to_sim",
                        "object_id": str(object_id),
                        "t_s": t_s,
                        "seq": int(command.get("seq", -1)),
                        "sim_time_ns": int(command.get("sim_time_ns", 0)),
                        "packet_hex": str(command.get("packet_hex", "")),
                        "age_ms": float(command.get("age_ms", 0.0)),
                        "stale": bool(command.get("stale", False)),
                        "decoded": {k: v for k, v in command.items() if k not in {"packet_hex"}},
                    }
                )
    return rows


def _summary(*, rows: list[dict[str, Any]], bridge_hist: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    directions = [str(row.get("direction", "")) for row in rows]
    command_rows = [row for row in rows if row.get("direction") == "cfs_to_sim"]
    stale_count = sum(1 for row in command_rows if bool(row.get("stale", False)))
    max_age_ms = max((float(row.get("age_ms", 0.0)) for row in command_rows), default=0.0)
    modes: dict[str, int] = {}
    for row in command_rows:
        mode = str(dict(row.get("decoded", {}) or {}).get("mode", "unknown"))
        modes[mode] = modes.get(mode, 0) + 1
    bridge_counts: dict[str, dict[str, Any]] = {}
    for object_id, events in sorted(dict(bridge_hist or {}).items()):
        cfs_events = [
            evt
            for evt in list(events or [])
            if isinstance(evt, dict)
            and isinstance(evt.get("bridge", {}), dict)
            and evt.get("bridge", {}).get("source") == "cfs_sil_bridge"
        ]
        if not cfs_events:
            continue
        last_bridge = dict(cfs_events[-1].get("bridge", {}) or {})
        bridge_counts[str(object_id)] = {
            "events": len(cfs_events),
            "sensor_tx_count": int(last_bridge.get("sensor_tx_count", 0)),
            "command_rx_count": int(last_bridge.get("command_rx_count", 0)),
            "decode_error_count": int(last_bridge.get("decode_error_count", 0)),
            "sequence_gap_count": int(last_bridge.get("sequence_gap_count", 0)),
            "out_of_order_count": int(last_bridge.get("out_of_order_count", 0)),
            "timestamp_mismatch_count": int(last_bridge.get("timestamp_mismatch_count", 0)),
            "last_command_age_ms": float(last_bridge.get("last_command_age_ms", 0.0)),
            "last_command_stale": bool(last_bridge.get("last_command_stale", False)),
            "execution_mode": str(last_bridge.get("execution_mode", "unknown")),
            "timeout_policy": str(last_bridge.get("timeout_policy", "unknown")),
            "command_validity_clock": str(last_bridge.get("command_validity_clock", "unknown")),
            "step_wait_count": int(last_bridge.get("step_wait_count", 0)),
            "step_timeout_count": int(last_bridge.get("step_timeout_count", 0)),
            "max_step_wait_ms": float(last_bridge.get("max_step_wait_ms", 0.0)),
        }
    return {
        "schema": "oel_cfs_sil_bridge_summary_v1",
        "packet_count": len(rows),
        "sensor_packet_count": directions.count("sim_to_cfs"),
        "command_packet_count": directions.count("cfs_to_sim"),
        "stale_command_packet_count": stale_count,
        "max_command_age_ms": max_age_ms,
        "command_modes": modes,
        "objects": bridge_counts,
    }


def _format_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# cFS SIL Bridge Summary",
        "",
        f"- Packet rows: `{int(summary.get('packet_count', 0))}`",
        f"- Sensor packets: `{int(summary.get('sensor_packet_count', 0))}`",
        f"- Command packets: `{int(summary.get('command_packet_count', 0))}`",
        f"- Stale command packets: `{int(summary.get('stale_command_packet_count', 0))}`",
        f"- Max command age: `{float(summary.get('max_command_age_ms', 0.0)):.3f} ms`",
        "",
        "## Command Modes",
        "",
    ]
    modes = dict(summary.get("command_modes", {}) or {})
    if modes:
        for mode, count in sorted(modes.items()):
            lines.append(f"- `{mode}`: `{int(count)}`")
    else:
        lines.append("- No command packets were received.")
    lines.extend(["", "## Objects", ""])
    objects = dict(summary.get("objects", {}) or {})
    if not objects:
        lines.append("- No cFS bridge object events were recorded.")
    for object_id, row in sorted(objects.items()):
        data = dict(row or {})
        lines.append(
            f"- `{object_id}`: events=`{int(data.get('events', 0))}`, "
            f"sensor_tx=`{int(data.get('sensor_tx_count', 0))}`, "
            f"command_rx=`{int(data.get('command_rx_count', 0))}`, "
            f"decode_errors=`{int(data.get('decode_error_count', 0))}`, "
            f"sequence_gaps=`{int(data.get('sequence_gap_count', 0))}`, "
            f"out_of_order=`{int(data.get('out_of_order_count', 0))}`, "
            f"time_mismatch=`{int(data.get('timestamp_mismatch_count', 0))}`, "
            f"last_age_ms=`{float(data.get('last_command_age_ms', 0.0)):.3f}`, "
            f"last_stale=`{bool(data.get('last_command_stale', False))}`, "
            f"execution=`{data.get('execution_mode', 'unknown')}`, "
            f"timeouts=`{int(data.get('step_timeout_count', 0))}`"
        )
    lines.append("")
    return "\n".join(lines)
