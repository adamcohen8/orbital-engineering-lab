from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC = b"RPO1"
VERSION = 1
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_EXECUTION_MODES = {"free_running", "paced", "deterministic_step"}
_TIMEOUT_POLICIES = {"error", "zero", "hold_last", "safe_mode"}

MSG_SIM_SENSOR_STATE = 0x1900
MSG_CFS_ACTUATOR_CMD = 0x1901

HEADER_FMT = "<4sHHIQ"  # magic, version, msg_id, seq, sim_time_ns
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SENSOR_FMT = "<B 3d 3d 4d 3d d 3d d"
CMD_FMT = "<B 3d 3d 3d I"
SENSOR_SIZE = struct.calcsize(SENSOR_FMT)
CMD_SIZE = struct.calcsize(CMD_FMT)


def _is_loopback_host(host: str) -> bool:
    return str(host or "").strip().lower() in _LOOPBACK_HOSTS


def _command_source_matches(addr: tuple[str, int], expected: tuple[str, int]) -> bool:
    actual_host, actual_port = str(addr[0]), int(addr[1])
    expected_host, expected_port = str(expected[0]), int(expected[1])
    if actual_port != expected_port:
        return False
    if actual_host == expected_host:
        return True
    return _is_loopback_host(actual_host) and _is_loopback_host(expected_host)


@dataclass
class SimSensorState:
    valid_flags: int
    pos_eci_km: np.ndarray
    vel_eci_km_s: np.ndarray
    quat_bn: np.ndarray
    omega_body_rad_s: np.ndarray
    mass_kg: float
    sun_dir_eci: np.ndarray
    density_kg_m3: float


@dataclass
class CfsActuatorCommand:
    mode: int
    thrust_eci_km_s2: np.ndarray
    torque_body_nm: np.ndarray
    wheel_torque_nm: np.ndarray
    valid_timeout_ms: int


def _pack_header(msg_id: int, seq: int, sim_time_ns: int) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, VERSION, int(msg_id), int(seq), int(sim_time_ns))


def _unpack_header(raw: bytes) -> tuple[int, int, int]:
    if len(raw) < HEADER_SIZE:
        raise ValueError("Packet too short for header.")
    magic, version, msg_id, seq, sim_time_ns = struct.unpack(HEADER_FMT, raw[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("Bad magic.")
    if version != VERSION:
        raise ValueError(f"Unsupported protocol version {version}.")
    return int(msg_id), int(seq), int(sim_time_ns)


def encode_sensor_packet(state: SimSensorState, seq: int, sim_time_ns: int) -> bytes:
    payload = struct.pack(
        SENSOR_FMT,
        int(state.valid_flags),
        *np.array(state.pos_eci_km, dtype=float).reshape(3),
        *np.array(state.vel_eci_km_s, dtype=float).reshape(3),
        *np.array(state.quat_bn, dtype=float).reshape(4),
        *np.array(state.omega_body_rad_s, dtype=float).reshape(3),
        float(state.mass_kg),
        *np.array(state.sun_dir_eci, dtype=float).reshape(3),
        float(state.density_kg_m3),
    )
    return _pack_header(MSG_SIM_SENSOR_STATE, seq, sim_time_ns) + payload


def decode_sensor_packet(raw: bytes) -> tuple[int, int, SimSensorState]:
    msg_id, seq, sim_time_ns = _unpack_header(raw)
    if msg_id != MSG_SIM_SENSOR_STATE:
        raise ValueError(f"Unexpected msg_id 0x{msg_id:04X}.")
    payload = raw[HEADER_SIZE:]
    if len(payload) != SENSOR_SIZE:
        raise ValueError(f"Invalid sensor payload size {len(payload)} != {SENSOR_SIZE}.")
    values = struct.unpack(SENSOR_FMT, payload)
    st = SimSensorState(
        valid_flags=int(values[0]),
        pos_eci_km=np.array(values[1:4], dtype=float),
        vel_eci_km_s=np.array(values[4:7], dtype=float),
        quat_bn=np.array(values[7:11], dtype=float),
        omega_body_rad_s=np.array(values[11:14], dtype=float),
        mass_kg=float(values[14]),
        sun_dir_eci=np.array(values[15:18], dtype=float),
        density_kg_m3=float(values[18]),
    )
    return seq, sim_time_ns, st


def encode_command_packet(cmd: CfsActuatorCommand, seq: int, sim_time_ns: int) -> bytes:
    payload = struct.pack(
        CMD_FMT,
        int(cmd.mode),
        *np.array(cmd.thrust_eci_km_s2, dtype=float).reshape(3),
        *np.array(cmd.torque_body_nm, dtype=float).reshape(3),
        *np.array(cmd.wheel_torque_nm, dtype=float).reshape(3),
        int(cmd.valid_timeout_ms),
    )
    return _pack_header(MSG_CFS_ACTUATOR_CMD, seq, sim_time_ns) + payload


def decode_command_packet(raw: bytes) -> tuple[int, int, CfsActuatorCommand]:
    msg_id, seq, sim_time_ns = _unpack_header(raw)
    if msg_id != MSG_CFS_ACTUATOR_CMD:
        raise ValueError(f"Unexpected msg_id 0x{msg_id:04X}.")
    payload = raw[HEADER_SIZE:]
    if len(payload) != CMD_SIZE:
        raise ValueError(f"Invalid command payload size {len(payload)} != {CMD_SIZE}.")
    values = struct.unpack(CMD_FMT, payload)
    cmd = CfsActuatorCommand(
        mode=int(values[0]),
        thrust_eci_km_s2=np.array(values[1:4], dtype=float),
        torque_body_nm=np.array(values[4:7], dtype=float),
        wheel_torque_nm=np.array(values[7:10], dtype=float),
        valid_timeout_ms=int(values[10]),
    )
    return seq, sim_time_ns, cmd


class CfsSilUdpBridge:
    """
    UDP bridge endpoint for simulator <-> cFS SIL coupling.

    - Sends SIM_SENSOR_STATE packets to cFS.
    - Receives CFS_ACTUATOR_CMD packets from cFS.
    """

    def __init__(
        self,
        local_bind: tuple[str, int],
        cfs_addr: tuple[str, int],
        *,
        allow_non_loopback: bool = False,
    ) -> None:
        if not bool(allow_non_loopback) and (
            not _is_loopback_host(str(local_bind[0])) or not _is_loopback_host(str(cfs_addr[0]))
        ):
            raise ValueError(
                "cFS SIL UDP bridge bind and destination hosts must be loopback by default. "
                "Set allow_non_loopback=True only for trusted isolated test networks."
            )
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(local_bind)
        self.sock.setblocking(False)
        self.cfs_addr = cfs_addr
        self.tx_seq = 0
        self.last_cmd_seq = -1
        self.last_cmd_sim_time_ns = 0
        self.last_cmd = CfsActuatorCommand(
            mode=0,
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            wheel_torque_nm=np.zeros(3),
            valid_timeout_ms=200,
        )
        self.last_cmd_rx_wall_time_ns = time.time_ns()
        self.last_sensor_seq = -1
        self.last_sensor_sim_time_ns = 0
        self.last_sensor_packet_hex = ""
        self.last_cmd_packet_hex = ""
        self.last_cmd_age_ms = 0.0
        self.last_cmd_stale = False
        self.decode_error_count = 0
        self.sequence_gap_count = 0
        self.out_of_order_count = 0
        self.timestamp_mismatch_count = 0

    def send_sensor(self, state: SimSensorState, sim_time_ns: int) -> bytes:
        pkt = encode_sensor_packet(state, seq=self.tx_seq, sim_time_ns=sim_time_ns)
        self.last_sensor_seq = int(self.tx_seq)
        self.last_sensor_sim_time_ns = int(sim_time_ns)
        self.last_sensor_packet_hex = pkt.hex()
        self.sock.sendto(pkt, self.cfs_addr)
        self.tx_seq += 1
        return pkt

    @staticmethod
    def _zero_command(valid_timeout_ms: int, *, mode: int = 0) -> CfsActuatorCommand:
        return CfsActuatorCommand(
            mode=int(mode),
            thrust_eci_km_s2=np.zeros(3),
            torque_body_nm=np.zeros(3),
            wheel_torque_nm=np.zeros(3),
            valid_timeout_ms=int(valid_timeout_ms),
        )

    @staticmethod
    def _valid_command(cmd: CfsActuatorCommand) -> bool:
        return bool(
            int(cmd.mode) in {0, 1, 2}
            and int(cmd.valid_timeout_ms) > 0
            and np.all(np.isfinite(cmd.thrust_eci_km_s2))
            and np.all(np.isfinite(cmd.torque_body_nm))
            and np.all(np.isfinite(cmd.wheel_torque_nm))
        )

    def _command_for_policy(self, policy: str) -> CfsActuatorCommand:
        if policy == "hold_last":
            return self.last_cmd
        if policy == "safe_mode":
            return self._zero_command(self.last_cmd.valid_timeout_ms, mode=2)
        return self._zero_command(self.last_cmd.valid_timeout_ms)

    def poll_command(
        self,
        *,
        current_sim_time_ns: int | None = None,
        stale_policy: str = "zero",
    ) -> CfsActuatorCommand:
        if stale_policy not in _TIMEOUT_POLICIES:
            raise ValueError(f"Unsupported stale command policy {stale_policy!r}.")
        while True:
            try:
                raw, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            if not _command_source_matches(addr, self.cfs_addr):
                self.decode_error_count += 1
                continue
            try:
                seq, sim_time_ns, cmd = decode_command_packet(raw)
            except ValueError:
                self.decode_error_count += 1
                continue
            if not self._valid_command(cmd):
                self.decode_error_count += 1
                continue
            if current_sim_time_ns is not None and int(sim_time_ns) > int(current_sim_time_ns):
                self.timestamp_mismatch_count += 1
                continue
            expected_seq = (int(self.last_cmd_seq) + 1) & 0xFFFFFFFF
            if self.last_cmd_seq >= 0 and int(seq) != expected_seq:
                if int(seq) <= int(self.last_cmd_seq) and not (int(self.last_cmd_seq) == 0xFFFFFFFF and int(seq) == 0):
                    self.out_of_order_count += 1
                    continue
                self.sequence_gap_count += 1
            self.last_cmd = cmd
            self.last_cmd_seq = seq
            self.last_cmd_sim_time_ns = sim_time_ns
            self.last_cmd_packet_hex = raw.hex()
            self.last_cmd_rx_wall_time_ns = time.time_ns()
        if current_sim_time_ns is None:
            age_ms = (time.time_ns() - self.last_cmd_rx_wall_time_ns) / 1e6
        else:
            age_ms = max(0, int(current_sim_time_ns) - int(self.last_cmd_sim_time_ns)) / 1e6
        self.last_cmd_age_ms = float(age_ms)
        self.last_cmd_stale = bool(age_ms > max(1, self.last_cmd.valid_timeout_ms))
        if self.last_cmd_stale:
            if stale_policy == "error":
                raise TimeoutError(
                    f"cFS command expired at simulation time: age={age_ms:.3f} ms, "
                    f"limit={self.last_cmd.valid_timeout_ms} ms"
                )
            return self._command_for_policy(stale_policy)
        return self.last_cmd

    def wait_for_command(
        self,
        *,
        sensor_sim_time_ns: int,
        timeout_s: float,
        poll_interval_s: float,
    ) -> tuple[CfsActuatorCommand, float]:
        started = time.monotonic()
        deadline = started + float(timeout_s)
        while True:
            cmd = self.poll_command(current_sim_time_ns=sensor_sim_time_ns, stale_policy="hold_last")
            if self.last_cmd_seq >= 0 and int(self.last_cmd_sim_time_ns) == int(sensor_sim_time_ns):
                return cmd, (time.monotonic() - started) * 1000.0
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    "Timed out waiting for a cFS actuator command matching sensor simulation time "
                    f"{int(sensor_sim_time_ns)} ns"
                )
            time.sleep(min(float(poll_interval_s), max(0.0, deadline - now)))


def _demo_sensor_state(t_s: float) -> SimSensorState:
    return SimSensorState(
        valid_flags=0x0F,
        pos_eci_km=np.array([6778.0 * np.cos(t_s / 5400.0), 6778.0 * np.sin(t_s / 5400.0), 0.0]),
        vel_eci_km_s=np.array([-7.6 * np.sin(t_s / 5400.0), 7.6 * np.cos(t_s / 5400.0), 0.0]),
        quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.zeros(3),
        mass_kg=300.0,
        sun_dir_eci=np.array([1.0, 0.0, 0.0]),
        density_kg_m3=1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="UDP bridge endpoint for cFS SIL integration.")
    parser.add_argument("--bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, default=50100)
    parser.add_argument("--cfs-host", type=str, default="127.0.0.1")
    parser.add_argument("--cfs-port", type=int, default=50101)
    parser.add_argument("--allow-non-loopback", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--demo", action="store_true", help="Run standalone packet loop with synthetic sensor state.")
    args = parser.parse_args()

    bridge = CfsSilUdpBridge(
        local_bind=(args.bind_host, int(args.bind_port)),
        cfs_addr=(args.cfs_host, int(args.cfs_port)),
        allow_non_loopback=bool(args.allow_non_loopback),
    )
    period_s = 1.0 / max(float(args.rate_hz), 1e-3)
    t0 = time.time()
    k = 0
    if args.demo:
        print(
            f"cFS SIL bridge demo running: bind={args.bind_host}:{args.bind_port} -> "
            f"cfs={args.cfs_host}:{args.cfs_port}, rate={args.rate_hz:.2f} Hz"
        )
    while args.demo:
        t_s = k * period_s
        sim_time_ns = int(t_s * 1e9)
        state = _demo_sensor_state(t_s)
        bridge.send_sensor(state, sim_time_ns=sim_time_ns)
        cmd = bridge.poll_command()
        if k % int(max(1, round(args.rate_hz))) == 0:
            print(
                f"t={t_s:8.2f}s seq={bridge.tx_seq:6d} cmd_mode={cmd.mode} "
                f"thrust=({cmd.thrust_eci_km_s2[0]:+.3e},{cmd.thrust_eci_km_s2[1]:+.3e},{cmd.thrust_eci_km_s2[2]:+.3e})"
            )
        k += 1
        next_t = t0 + k * period_s
        sleep_s = next_t - time.time()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
