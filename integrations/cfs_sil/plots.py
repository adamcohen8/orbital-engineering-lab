from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.utils.figure_size import cap_figsize

FIGURE_IDS = ("cfs_attitude_sensor", "cfs_actuator_commands")


def bridge_io_histories(bridge_hist: dict[str, list[dict[str, Any]]] | None) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    if not isinstance(bridge_hist, dict):
        return out
    for oid, events in bridge_hist.items():
        if not isinstance(events, list) or not events:
            continue
        times: list[float] = []
        sensor_quat: list[np.ndarray] = []
        sensor_omega: list[np.ndarray] = []
        cmd_torque: list[np.ndarray] = []
        cmd_wheel: list[np.ndarray] = []
        cmd_mode: list[float] = []
        for evt in events:
            if not isinstance(evt, dict):
                continue
            bridge = evt.get("bridge", {})
            if not isinstance(bridge, dict):
                continue
            sensor = bridge.get("last_sensor_state", {})
            command = bridge.get("last_actuator_command", {})
            if not isinstance(sensor, dict):
                sensor = {}
            if not isinstance(command, dict):
                command = {}
            times.append(float(evt.get("t_s", np.nan)))
            sensor_quat.append(_bridge_vector(sensor.get("quat_bn"), size=4))
            sensor_omega.append(_bridge_vector(sensor.get("omega_body_rad_s"), size=3))
            cmd_torque.append(_bridge_vector(command.get("torque_body_nm"), size=3))
            cmd_wheel.append(_bridge_vector(command.get("wheel_torque_nm"), size=3))
            cmd_mode.append(float(command.get("mode", np.nan)))
        if not times:
            continue
        out[str(oid)] = {
            "t_s": np.array(times, dtype=float),
            "sensor_quat_bn": np.vstack(sensor_quat) if sensor_quat else np.zeros((0, 4), dtype=float),
            "sensor_omega_body_rad_s": np.vstack(sensor_omega) if sensor_omega else np.zeros((0, 3), dtype=float),
            "cmd_torque_body_nm": np.vstack(cmd_torque) if cmd_torque else np.zeros((0, 3), dtype=float),
            "cmd_wheel_torque_nm": np.vstack(cmd_wheel) if cmd_wheel else np.zeros((0, 3), dtype=float),
            "cmd_mode": np.array(cmd_mode, dtype=float),
        }
    return out


def plot_bridge_outputs(
    *,
    figure_ids: list[str],
    bridge_hist: dict[str, list[dict[str, Any]]] | None,
    outdir: Path,
    mode: str,
    dpi: int,
) -> dict[str, str]:
    requested = set(str(item) for item in figure_ids)
    if not requested.intersection(FIGURE_IDS):
        return {}

    import matplotlib.pyplot as plt

    out: dict[str, str] = {}
    bridge_series = bridge_io_histories(bridge_hist)
    for oid, series in bridge_series.items():
        tb = np.array(series.get("t_s", []), dtype=float).reshape(-1)
        if tb.size == 0:
            continue
        if "cfs_attitude_sensor" in requested:
            q = np.array(series.get("sensor_quat_bn", []), dtype=float)
            w = np.array(series.get("sensor_omega_body_rad_s", []), dtype=float)
            if q.ndim == 2 and w.ndim == 2 and (np.any(np.isfinite(q)) or np.any(np.isfinite(w))):
                fig, axes = plt.subplots(2, 1, figsize=cap_figsize(11, 7), sharex=True)
                n_q = min(tb.size, q.shape[0])
                for i, label in enumerate(("q0", "q1", "q2", "q3")):
                    axes[0].plot(tb[:n_q], q[:n_q, i], linewidth=1.0, label=label)
                axes[0].set_ylabel("Quaternion")
                axes[0].set_title(f"cFS Attitude Sensor Packet ({oid})")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="best")
                n_w = min(tb.size, w.shape[0])
                for i, label in enumerate(("wx", "wy", "wz")):
                    axes[1].plot(tb[:n_w], w[:n_w, i], linewidth=1.0, label=label)
                axes[1].set_xlabel("Time (s)")
                axes[1].set_ylabel("Body Rate (rad/s)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="best")
                fig.tight_layout()
                path = outdir / f"{oid}_cfs_attitude_sensor.png"
                if mode in ("save", "both"):
                    fig.savefig(path, dpi=dpi)
                    out[f"{oid}_cfs_attitude_sensor"] = str(path)
                if mode in ("interactive", "both"):
                    plt.show(block=False)
                else:
                    plt.close(fig)
        if "cfs_actuator_commands" in requested:
            torque = np.array(series.get("cmd_torque_body_nm", []), dtype=float)
            wheel = np.array(series.get("cmd_wheel_torque_nm", []), dtype=float)
            modes = np.array(series.get("cmd_mode", []), dtype=float).reshape(-1)
            if torque.ndim == 2 and wheel.ndim == 2 and (np.any(np.isfinite(torque)) or np.any(np.isfinite(wheel))):
                fig, axes = plt.subplots(3, 1, figsize=cap_figsize(11, 8), sharex=True)
                n_torque = min(tb.size, torque.shape[0])
                for i, label in enumerate(("tx", "ty", "tz")):
                    axes[0].plot(tb[:n_torque], torque[:n_torque, i], linewidth=1.0, label=label)
                axes[0].set_ylabel("Torque (N-m)")
                axes[0].set_title(f"cFS Actuator Command Packet ({oid})")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="best")
                n_wheel = min(tb.size, wheel.shape[0])
                for i, label in enumerate(("wx", "wy", "wz")):
                    axes[1].plot(tb[:n_wheel], wheel[:n_wheel, i], linewidth=1.0, label=label)
                axes[1].set_ylabel("Wheel Torque (N-m)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="best")
                n_mode = min(tb.size, modes.size)
                axes[2].step(tb[:n_mode], modes[:n_mode], where="post", linewidth=1.0)
                axes[2].set_xlabel("Time (s)")
                axes[2].set_ylabel("Mode")
                axes[2].grid(True, alpha=0.3)
                fig.tight_layout()
                path = outdir / f"{oid}_cfs_actuator_commands.png"
                if mode in ("save", "both"):
                    fig.savefig(path, dpi=dpi)
                    out[f"{oid}_cfs_actuator_commands"] = str(path)
                if mode in ("interactive", "both"):
                    plt.show(block=False)
                else:
                    plt.close(fig)
    return out


def _bridge_vector(value: Any, *, size: int) -> np.ndarray:
    arr = np.array(value if value is not None else [], dtype=float).reshape(-1)
    out = np.full(size, np.nan, dtype=float)
    n = min(size, arr.size)
    if n > 0:
        out[:n] = arr[:n]
    return out
