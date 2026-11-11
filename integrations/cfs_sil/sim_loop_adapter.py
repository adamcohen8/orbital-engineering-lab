from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from integrations.cfs_sil.python_bridge import CfsSilUdpBridge, SimSensorState
from sim.core.models import Command
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.presets import BASIC_SATELLITE, build_sim_object_from_presets


@dataclass(frozen=True)
class CfsSilLoopConfig:
    dt_s: float = 0.01
    duration_s: float = 300.0
    bind_host: str = "127.0.0.1"
    bind_port: int = 50100
    cfs_host: str = "127.0.0.1"
    cfs_port: int = 50101
    atmosphere_model: str = "ussa1976"
    orbit_radius_km: float = 6778.0
    phase_rad: float = 0.0
    enable_disturbances: bool = True


@dataclass
class CfsSilLoopResult:
    time_s: np.ndarray
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    commanded_thrust_eci_km_s2: np.ndarray
    commanded_torque_body_nm: np.ndarray
    bridge_cmd_mode: np.ndarray


def run_single_satellite_cfs_sil_loop(cfg: CfsSilLoopConfig) -> CfsSilLoopResult:
    sat = build_sim_object_from_presets(
        object_id="cfs_sil_sat",
        dt_s=cfg.dt_s,
        satellite=BASIC_SATELLITE,
        orbit_radius_km=cfg.orbit_radius_km,
        phase_rad=cfg.phase_rad,
        enable_disturbances=cfg.enable_disturbances,
        enable_attitude_knowledge=True,
    )
    bridge = CfsSilUdpBridge(
        local_bind=(cfg.bind_host, int(cfg.bind_port)),
        cfs_addr=(cfg.cfs_host, int(cfg.cfs_port)),
    )

    steps = int(np.ceil(cfg.duration_s / cfg.dt_s))
    t = np.zeros(steps + 1)
    r = np.zeros((steps + 1, 3))
    v = np.zeros((steps + 1, 3))
    q = np.zeros((steps + 1, 4))
    w = np.zeros((steps + 1, 3))
    thrust_cmd = np.zeros((steps + 1, 3))
    torque_cmd = np.zeros((steps + 1, 3))
    mode_cmd = np.zeros(steps + 1, dtype=np.int64)

    env = {
        "sun_dir_eci": np.array([1.0, 0.1, 0.0], dtype=float),
        "atmosphere_model": cfg.atmosphere_model,
    }

    for k in range(steps + 1):
        t_now = sat.truth.t_s
        t[k] = t_now
        r[k, :] = sat.truth.position_eci_km
        v[k, :] = sat.truth.velocity_eci_km_s
        q[k, :] = sat.truth.attitude_quat_bn
        w[k, :] = sat.truth.angular_rate_body_rad_s

        rho = density_from_model(cfg.atmosphere_model, sat.truth.position_eci_km, t_now, env=env)
        sensor_pkt = SimSensorState(
            valid_flags=0x0F,
            pos_eci_km=sat.truth.position_eci_km,
            vel_eci_km_s=sat.truth.velocity_eci_km_s,
            quat_bn=sat.truth.attitude_quat_bn,
            omega_body_rad_s=sat.truth.angular_rate_body_rad_s,
            mass_kg=sat.truth.mass_kg,
            sun_dir_eci=np.array(env["sun_dir_eci"], dtype=float),
            density_kg_m3=float(rho),
        )
        bridge.send_sensor(sensor_pkt, sim_time_ns=int(t_now * 1e9))
        cmd_in = bridge.poll_command()
        mode_cmd[k] = int(cmd_in.mode)
        thrust_cmd[k, :] = np.array(cmd_in.thrust_eci_km_s2, dtype=float)
        torque_cmd[k, :] = np.array(cmd_in.torque_body_nm, dtype=float)

        if k == steps:
            break

        # Keep estimator/sensor loop active to stay aligned with existing sim architecture.
        meas = sat.sensor.measure(sat.truth, env=env, t_s=t_now + cfg.dt_s)
        sat.belief = sat.estimator.update(sat.belief, meas, t_s=t_now + cfg.dt_s)

        sim_cmd = Command(
            thrust_eci_km_s2=np.array(cmd_in.thrust_eci_km_s2, dtype=float),
            torque_body_nm=np.array(cmd_in.torque_body_nm, dtype=float),
            mode_flags={"source": "cfs_sil_bridge", "bridge_mode": int(cmd_in.mode)},
        )
        applied = sat.actuator.apply(sim_cmd, sat.limits, cfg.dt_s)
        sat.truth = sat.dynamics.step(sat.truth, applied, env=env, dt_s=cfg.dt_s)

    return CfsSilLoopResult(
        time_s=t,
        position_eci_km=r,
        velocity_eci_km_s=v,
        attitude_quat_bn=q,
        angular_rate_body_rad_s=w,
        commanded_thrust_eci_km_s2=thrust_cmd,
        commanded_torque_body_nm=torque_cmd,
        bridge_cmd_mode=mode_cmd,
    )
