from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from integrations.cfs_sil.python_bridge import (
    CfsActuatorCommand,
    _is_loopback_host,
    decode_sensor_packet,
    encode_command_packet,
)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3 = _normalize_quat(q1)
    b0, b1, b2, b3 = _normalize_quat(q2)
    return _normalize_quat(
        np.array(
            [
                a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
            ],
            dtype=float,
        )
    )


def _att_hold_torque(
    *,
    sensor_quat_bn: np.ndarray,
    sensor_omega_body_rad_s: np.ndarray,
    desired_quat_bn: np.ndarray,
    kp: float,
    kd: float,
    max_torque_nm: float,
) -> np.ndarray:
    q_err = _quat_multiply(_quat_conjugate(desired_quat_bn), sensor_quat_bn)
    if float(q_err[0]) < 0.0:
        q_err *= -1.0
    torque = -float(kp) * q_err[1:4] - float(kd) * np.array(sensor_omega_body_rad_s, dtype=float).reshape(3)
    n = float(np.linalg.norm(torque))
    if n > float(max_torque_nm) > 0.0:
        torque *= float(max_torque_nm) / n
    return torque


def _build_command(
    *,
    sensor_vel_eci: np.ndarray,
    sensor_quat_bn: np.ndarray,
    sensor_omega_body_rad_s: np.ndarray,
    desired_quat_bn: np.ndarray,
    mode: str,
    accel_mag_km_s2: float,
    valid_timeout_ms: int,
    att_kp: float,
    att_kd: float,
    max_torque_nm: float,
) -> CfsActuatorCommand:
    thrust = np.zeros(3)
    torque = np.zeros(3)
    wheel = np.zeros(3)

    if mode == "hold":
        pass
    elif mode == "prograde":
        thrust = accel_mag_km_s2 * _unit(sensor_vel_eci)
    elif mode == "retrograde":
        thrust = -accel_mag_km_s2 * _unit(sensor_vel_eci)
    elif mode == "att_damp":
        torque = -float(att_kd) * np.array(sensor_omega_body_rad_s, dtype=float).reshape(3)
        n = float(np.linalg.norm(torque))
        if n > float(max_torque_nm) > 0.0:
            torque *= float(max_torque_nm) / n
        wheel = torque.copy()
    elif mode == "att_hold":
        torque = _att_hold_torque(
            sensor_quat_bn=sensor_quat_bn,
            sensor_omega_body_rad_s=sensor_omega_body_rad_s,
            desired_quat_bn=desired_quat_bn,
            kp=att_kp,
            kd=att_kd,
            max_torque_nm=max_torque_nm,
        )
        wheel = torque.copy()
    else:
        pass

    return CfsActuatorCommand(
        mode=1,
        thrust_eci_km_s2=thrust,
        torque_body_nm=torque,
        wheel_torque_nm=wheel,
        valid_timeout_ms=int(valid_timeout_ms),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock cFS UDP endpoint for SIL loop testing.")
    parser.add_argument("--bind-host", type=str, default="127.0.0.1", help="Mock endpoint bind host.")
    parser.add_argument("--bind-port", type=int, default=50101, help="Mock endpoint bind port (receives sensors).")
    parser.add_argument("--sim-host", type=str, default="127.0.0.1", help="Simulator host to send commands to.")
    parser.add_argument("--sim-port", type=int, default=50100, help="Simulator port to send commands to.")
    parser.add_argument("--mode", choices=["hold", "prograde", "retrograde", "att_damp", "att_hold"], default="hold")
    parser.add_argument(
        "--accel-mag-km-s2", type=float, default=2e-6, help="Thrust accel magnitude for pro/retrograde."
    )
    parser.add_argument("--att-kp", type=float, default=0.10, help="Quaternion vector proportional gain for att_hold.")
    parser.add_argument("--att-kd", type=float, default=0.05, help="Body-rate damping gain for att_hold.")
    parser.add_argument("--max-torque-nm", type=float, default=0.05, help="Torque norm limit for att_hold.")
    parser.add_argument(
        "--desired-quat-bn",
        type=float,
        nargs=4,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Desired body-to-navigation quaternion for att_hold.",
    )
    parser.add_argument("--valid-timeout-ms", type=int, default=500)
    parser.add_argument("--allow-non-loopback", action="store_true")
    args = parser.parse_args()

    if not bool(args.allow_non_loopback) and not _is_loopback_host(args.bind_host):
        raise ValueError(
            "Mock cFS endpoint bind host must be loopback by default. "
            "Pass --allow-non-loopback only for trusted isolated test networks."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_host, int(args.bind_port)))
    sock.settimeout(1.0)

    sim_addr = (args.sim_host, int(args.sim_port))
    tx_seq = 0
    rx_count = 0
    t_last_print = time.time()

    print(
        f"Mock cFS endpoint listening on {args.bind_host}:{args.bind_port}, "
        f"sending commands to {args.sim_host}:{args.sim_port}, mode={args.mode}"
    )
    while True:
        try:
            raw, _addr = sock.recvfrom(4096)
        except socket.timeout:
            continue

        try:
            rx_seq, sim_time_ns, sensor = decode_sensor_packet(raw)
        except ValueError:
            continue

        cmd = _build_command(
            sensor_vel_eci=np.array(sensor.vel_eci_km_s, dtype=float),
            sensor_quat_bn=np.array(sensor.quat_bn, dtype=float),
            sensor_omega_body_rad_s=np.array(sensor.omega_body_rad_s, dtype=float),
            desired_quat_bn=np.array(args.desired_quat_bn, dtype=float),
            mode=args.mode,
            accel_mag_km_s2=float(args.accel_mag_km_s2),
            valid_timeout_ms=int(args.valid_timeout_ms),
            att_kp=float(args.att_kp),
            att_kd=float(args.att_kd),
            max_torque_nm=float(args.max_torque_nm),
        )
        pkt = encode_command_packet(cmd, seq=tx_seq, sim_time_ns=sim_time_ns)
        sock.sendto(pkt, sim_addr)
        tx_seq += 1
        rx_count += 1

        now = time.time()
        if now - t_last_print >= 1.0:
            print(
                f"rx={rx_count:6d} rx_seq={rx_seq:6d} tx_seq={tx_seq:6d} "
                f"sim_t={sim_time_ns / 1e9:8.2f}s thrust=({cmd.thrust_eci_km_s2[0]:+.2e},"
                f"{cmd.thrust_eci_km_s2[1]:+.2e},{cmd.thrust_eci_km_s2[2]:+.2e})"
            )
            t_last_print = now


if __name__ == "__main__":
    main()
