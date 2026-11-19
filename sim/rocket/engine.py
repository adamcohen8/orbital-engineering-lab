from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sim.dynamics.attitude.rigid_body import propagate_attitude_exponential_map
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.frames import frame_context_from_environment, transform_position, transform_state
from sim.dynamics.orbit.propagator import OrbitPropagator, drag_plugin, j2_plugin, j3_plugin, j4_plugin, srp_plugin
from sim.numeric_backend import normalize_numeric_backend
from sim.rocket.aero import RocketAeroConfig, compute_aero_loads, compute_aero_state
from sim.rocket.landing_gear import GearContactModel, gear_evidence
from sim.rocket.landing_step import advance_held_gear, advance_landing
from sim.rocket.models import (
    GuidanceCommand,
    RocketGuidanceLaw,
    RocketSimConfig,
    RocketSimResult,
    RocketState,
    RocketVehicleConfig,
)
from sim.rocket.navigation import _clip_scalar, _cross3, _norm3, rocket_air_relative_state_eci_m_s
from sim.rocket.touchdown import RocketTouchdown, locate_touchdown
from sim.utils.geodesy import ecef_to_geodetic_deg_km, geodetic_to_ecef_km
from sim.utils.quaternion import normalize_quaternion, quaternion_to_dcm_bn

G0_M_S2 = 9.80665
P0_SEA_LEVEL_PA = 101325.0


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(_norm3(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _launch_position_velocity_eci(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    t_s: float,
    *,
    environment: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r_ecef = geodetic_to_ecef_km(lat_deg=lat_deg, lon_deg=lon_deg, alt_km=alt_km)
    frame_context = frame_context_from_environment(dict(environment or {}))
    # The pad is stationary in ECEF.  Transforming the complete state applies
    # the configured epoch/EOP rotation and its derivative consistently.
    return transform_state(
        r_ecef,
        np.zeros(3, dtype=float),
        "ecef",
        "eci",
        t_s=t_s,
        context=frame_context,
    )


def _geodetic_state_from_eci(
    r_eci_km: np.ndarray,
    t_s: float,
    jd_utc_start: float | None = None,
    frame_context=None,
) -> tuple[float, float, float]:
    frame_context = frame_context or frame_context_from_environment({"jd_utc_start": jd_utc_start})
    r_ecef = transform_position(
        np.asarray(r_eci_km, dtype=float).reshape(3),
        "eci",
        "ecef",
        t_s=t_s,
        context=frame_context,
    )
    return ecef_to_geodetic_deg_km(r_ecef)


def _vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(np.asarray(a, dtype=float).reshape(3))
    ub = _unit(np.asarray(b, dtype=float).reshape(3))
    if _norm3(ua) <= 0.0 or _norm3(ub) <= 0.0:
        return 0.0
    return float(np.rad2deg(np.arccos(_clip_scalar(float(np.dot(ua, ub)), -1.0, 1.0))))


def _limit_vector_cone(v: np.ndarray, axis: np.ndarray, max_angle_rad: float) -> np.ndarray:
    u = _unit(v)
    a = _unit(axis)
    if _norm3(u) <= 0.0:
        return a
    if max_angle_rad <= 0.0:
        return a
    angle = float(np.arccos(_clip_scalar(float(np.dot(u, a)), -1.0, 1.0)))
    if angle <= max_angle_rad:
        return u
    lateral = u - float(np.dot(u, a)) * a
    lateral = _unit(lateral)
    return _unit(np.cos(max_angle_rad) * a + np.sin(max_angle_rad) * lateral)


def _step_tvc_vector(
    current_body: np.ndarray,
    target_body: np.ndarray,
    nominal_axis_body: np.ndarray,
    dt_s: float,
    sim_cfg: RocketSimConfig,
) -> np.ndarray:
    current = _unit(current_body)
    nominal = _unit(nominal_axis_body)
    target = _limit_vector_cone(target_body, nominal, np.deg2rad(float(sim_cfg.tvc_max_gimbal_deg)))
    alpha = float(1.0 - np.exp(-dt_s / max(sim_cfg.tvc_time_constant_s, 1e-9)))
    blended = _unit((1.0 - alpha) * current + alpha * target)
    if sim_cfg.tvc_rate_limit_deg_s <= 0.0:
        return blended
    max_step = float(np.deg2rad(sim_cfg.tvc_rate_limit_deg_s) * dt_s)
    step_angle = float(np.arccos(_clip_scalar(float(np.dot(current, blended)), -1.0, 1.0)))
    if step_angle <= max_step or step_angle <= 1e-12:
        return blended
    beta = max_step / step_angle
    return _unit((1.0 - beta) * current + beta * blended)


def _stage_engine_perf(stage, pressure_pa: float, numeric_backend: str = "rust") -> tuple[float, float]:
    backend = normalize_numeric_backend(numeric_backend, error_message='numeric_backend must be python or rust')
    thrust_sl = float(stage.sea_level_thrust_n if stage.sea_level_thrust_n is not None else stage.max_thrust_n)
    thrust_vac = float(stage.vacuum_thrust_n if stage.vacuum_thrust_n is not None else stage.max_thrust_n)
    isp_sl = float(stage.sea_level_isp_s if stage.sea_level_isp_s is not None else stage.isp_s)
    isp_vac = float(stage.vacuum_isp_s if stage.vacuum_isp_s is not None else stage.isp_s)
    if backend == "rust":
        from sim.rust_vehicle_backend import rocket_stage_engine_perf

        return rocket_stage_engine_perf(
            pressure_pa=pressure_pa,
            sea_level_thrust_n=thrust_sl,
            vacuum_thrust_n=thrust_vac,
            sea_level_isp_s=isp_sl,
            vacuum_isp_s=isp_vac,
        )
    p = float(_clip_scalar(pressure_pa, 0.0, P0_SEA_LEVEL_PA))
    sea_w = p / P0_SEA_LEVEL_PA
    vac_w = 1.0 - sea_w
    thrust_n = sea_w * thrust_sl + vac_w * thrust_vac
    isp_s = sea_w * isp_sl + vac_w * isp_vac
    return float(max(thrust_n, 0.0)), float(max(isp_s, 1e-9))


def _initial_attitude_quaternion(r_eci_km: np.ndarray, azimuth_deg: float) -> np.ndarray:
    r_hat = _unit(r_eci_km)
    k = np.array([0.0, 0.0, 1.0], dtype=float)
    east = _unit(_cross3(k, r_hat))
    if _norm3(east) <= 0.0:
        east = np.array([0.0, 1.0, 0.0])
    north = _unit(_cross3(r_hat, east))
    az = np.deg2rad(azimuth_deg)
    # body +X along launch axis, initially near radial with azimuth yaw bias.
    x_b = _unit(
        np.cos(np.deg2rad(1.0)) * r_hat + np.sin(np.deg2rad(1.0)) * (_unit(np.cos(az) * north + np.sin(az) * east))
    )
    y_b = _unit(_cross3(k, x_b))
    if _norm3(y_b) <= 0.0:
        y_b = _unit(_cross3(np.array([0.0, 1.0, 0.0]), x_b))
    z_b = _unit(_cross3(x_b, y_b))
    c_bn = np.vstack((x_b, y_b, z_b))
    from sim.utils.quaternion import dcm_to_quaternion_bn

    return dcm_to_quaternion_bn(c_bn)


def _orbital_elements_basic(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float) -> tuple[float, float]:
    r = float(_norm3(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    if abs(eps) < 1e-14:
        a = np.inf
    else:
        a = -mu_km3_s2 / (2.0 * eps)
    h = _cross3(r_km, v_km_s)
    e_vec = _cross3(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(_norm3(e_vec))
    return float(a), e


@dataclass
class RocketAscentSimulator:
    sim_cfg: RocketSimConfig
    vehicle_cfg: RocketVehicleConfig
    guidance: RocketGuidanceLaw

    def __post_init__(self) -> None:
        self.gear_model = None
        if self.vehicle_cfg.landing_gear is not None:
            if self.sim_cfg.touchdown is None or self.sim_cfg.touchdown.contact_height_m != 0.0:
                raise ValueError("Landing gear requires a touchdown policy with zero contact_height_m.")
            if self.sim_cfg.attitude_mode.strip().lower() != "dynamic":
                raise ValueError("Landing gear requires dynamic rocket attitude.")
            self.gear_model = GearContactModel(self.vehicle_cfg.landing_gear, self.sim_cfg)
        stages = self.vehicle_cfg.stack.stages
        if len(stages) == 0:
            raise ValueError("vehicle_cfg.stack must contain at least one stage.")
        self._stage_dry = np.array([s.dry_mass_kg for s in stages], dtype=float)
        self._stage_prop0 = np.array([s.propellant_mass_kg for s in stages], dtype=float)
        self._stage_thrust = np.array([s.max_thrust_n for s in stages], dtype=float)
        self._stage_isp = np.array([s.isp_s for s in stages], dtype=float)
        self._stage_area_ref_m2 = np.array(
            [np.pi * 0.25 * float(s.diameter_m) * float(s.diameter_m) for s in stages], dtype=float
        )
        self._stage_ref_length_m = np.array([float(s.length_m) for s in stages], dtype=float)
        self._base_aero_ref_length_m = float(max(self.sim_cfg.aero.reference_length_m, 1e-9))
        if self.sim_cfg.area_ref_m2 is None:
            d = float(stages[0].diameter_m)
            self._area_ref_m2 = np.pi * 0.25 * d * d
        else:
            self._area_ref_m2 = float(self.sim_cfg.area_ref_m2)

        plugins = []
        if self.sim_cfg.enable_j2:
            plugins.append(j2_plugin)
        if self.sim_cfg.enable_j3:
            plugins.append(j3_plugin)
        if self.sim_cfg.enable_j4:
            plugins.append(j4_plugin)
        if self.sim_cfg.enable_drag and not self.sim_cfg.aero.enabled:
            plugins.append(drag_plugin)
        if self.sim_cfg.enable_srp:
            plugins.append(srp_plugin)
        self._propagator = OrbitPropagator(
            integrator="rk4",
            plugins=plugins,
            numeric_backend=self.sim_cfg.numeric_backend,
        )

    def _resolve_aero_config_for_stage(self, stage_i: int) -> RocketAeroConfig:
        cfg = self.sim_cfg.aero
        if not cfg.enabled:
            return cfg
        idx = int(_clip_scalar(stage_i, 0, len(self._stage_ref_length_m) - 1))

        # Optional global override keeps area fixed regardless of stage.
        area_m2 = (
            float(self.sim_cfg.area_ref_m2) if self.sim_cfg.area_ref_m2 is not None else float(cfg.reference_area_m2)
        )
        ref_len_m = float(cfg.reference_length_m)
        cp_offset = np.array(cfg.cp_offset_body_m, dtype=float).reshape(3)
        if self.sim_cfg.use_stagewise_aero_geometry:
            if self.sim_cfg.area_ref_m2 is None:
                area_m2 = float(self._stage_area_ref_m2[idx])
            ref_len_m = float(max(self._stage_ref_length_m[idx], 1e-9))
            scale = ref_len_m / self._base_aero_ref_length_m
            cp_offset = cp_offset * scale
        return replace(cfg, reference_area_m2=area_m2, reference_length_m=ref_len_m, cp_offset_body_m=cp_offset)

    def initial_state(self) -> RocketState:
        r0, v0 = _launch_position_velocity_eci(
            lat_deg=self.sim_cfg.launch_lat_deg,
            lon_deg=self.sim_cfg.launch_lon_deg,
            alt_km=self.sim_cfg.launch_alt_km,
            t_s=0.0,
            environment=self.sim_cfg.atmosphere_env,
        )
        q0 = _initial_attitude_quaternion(r0, self.sim_cfg.launch_azimuth_deg)
        mass0 = float(np.sum(self._stage_dry + self._stage_prop0) + self.vehicle_cfg.payload_mass_kg)
        return RocketState(
            t_s=0.0,
            position_eci_km=r0,
            velocity_eci_km_s=v0,
            attitude_quat_bn=q0,
            angular_rate_body_rad_s=np.zeros(3),
            mass_kg=mass0,
            active_stage_index=0,
            stage_prop_remaining_kg=self._stage_prop0.copy(),
            payload_attached=True,
            thrust_vector_body=_unit(np.asarray(self.vehicle_cfg.thrust_axis_body, dtype=float)),
            landing_legs={} if self.gear_model is None else self.gear_model.initial_legs(),
        )

    def hold_on_launch_pad(self, state: RocketState, *, t_s: float) -> RocketState:
        """Advance a prelaunch state while keeping it fixed to the rotating launch pad."""
        advance_held_gear(self, state, float(t_s))
        r_eci, v_eci = _launch_position_velocity_eci(
            lat_deg=self.sim_cfg.launch_lat_deg,
            lon_deg=self.sim_cfg.launch_lon_deg,
            alt_km=self.sim_cfg.launch_alt_km,
            t_s=float(t_s),
            environment=self.sim_cfg.atmosphere_env,
        )
        state.position_eci_km = r_eci
        state.velocity_eci_km_s = v_eci
        state.attitude_quat_bn = _initial_attitude_quaternion(r_eci, self.sim_cfg.launch_azimuth_deg)
        state.angular_rate_body_rad_s = np.zeros(3, dtype=float)
        state.t_s = float(t_s)
        return state

    def run(self, state0: RocketState | None = None) -> RocketSimResult:
        state = self.initial_state() if state0 is None else state0.copy()
        dt = self.sim_cfg.dt_s
        steps = int(np.ceil(self.sim_cfg.max_time_s / dt))

        t = np.zeros(steps + 1)
        r = np.zeros((steps + 1, 3))
        v = np.zeros((steps + 1, 3))
        q = np.zeros((steps + 1, 4))
        w = np.zeros((steps + 1, 3))
        m = np.zeros(steps + 1)
        stg = np.zeros(steps + 1, dtype=int)
        thr_cmd = np.zeros(steps + 1)
        thrust_n = np.zeros(steps + 1)
        alt = np.zeros(steps + 1)
        ecc = np.zeros(steps + 1)
        sma = np.zeros(steps + 1)
        q_dyn = np.zeros(steps + 1)
        mach = np.zeros(steps + 1)
        lat_deg = np.zeros(steps + 1)
        lon_deg = np.zeros(steps + 1)
        wind_body_m_s = np.zeros((steps + 1, 3))
        tvc_gimbal_deg = np.zeros(steps + 1)
        alpha_deg = np.zeros(steps + 1)
        beta_deg = np.zeros(steps + 1)
        cd = np.zeros(steps + 1)
        aero_force_n = np.zeros(steps + 1)
        aero_moment_nm = np.zeros(steps + 1)

        inserted = False
        insertion_time = None
        terminated_early = False
        termination_reason = None
        termination_time = None
        insertion_hold_counter = 0.0
        hold_needed = self.sim_cfg.insertion_hold_time_s

        def _record_sample(idx: int, sample_state: RocketState) -> None:
            t[idx] = sample_state.t_s
            r[idx, :] = sample_state.position_eci_km
            v[idx, :] = sample_state.velocity_eci_km_s
            q[idx, :] = sample_state.attitude_quat_bn
            w[idx, :] = sample_state.angular_rate_body_rad_s
            m[idx] = sample_state.mass_kg
            stg[idx] = sample_state.active_stage_index
            lat_now, lon_now, alt_now = _geodetic_state_from_eci(
                sample_state.position_eci_km,
                sample_state.t_s,
                jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
                frame_context=(
                    frame_context_from_environment(self.sim_cfg.atmosphere_env)
                    if self.sim_cfg.touchdown is not None else None
                ),
            )
            alt[idx] = float(
                alt_now
                if self.sim_cfg.use_wgs84_geodesy
                else _norm3(sample_state.position_eci_km) - EARTH_RADIUS_KM
            )
            lat_deg[idx] = float(lat_now)
            lon_deg[idx] = float(lon_now)
            a_km, e_k = _orbital_elements_basic(
                sample_state.position_eci_km, sample_state.velocity_eci_km_s, EARTH_MU_KM3_S2
            )
            sma[idx] = a_km
            ecc[idx] = e_k

        for k in range(steps + 1):
            _record_sample(k, state)
            if k == steps:
                break

            cmd = self.guidance.command(state, self.sim_cfg, self.vehicle_cfg)
            throttle = _clamp(float(cmd.throttle), 0.0, 1.0)
            interval = dt if self.gear_model is None else min(dt, self.sim_cfg.max_time_s - k * dt)
            trial = self.step(state, cmd, dt_s=interval)
            touchdown = self.resolve_touchdown(state, trial, cmd)
            state = trial if touchdown is None else touchdown.state
            _record_sample(k + 1, state)
            sample_idx = k + 1
            thr_cmd[sample_idx] = throttle
            thrust_n[sample_idx] = float(state._last_step_thrust_n) if hasattr(state, "_last_step_thrust_n") else 0.0
            q_dyn[sample_idx] = float(getattr(state, "_last_step_q_dyn_pa", 0.0))
            mach[sample_idx] = float(getattr(state, "_last_step_mach", 0.0))
            wind_body_m_s[sample_idx, :] = np.array(getattr(state, "_last_step_wind_body_m_s", np.zeros(3)), dtype=float)
            tvc_gimbal_deg[sample_idx] = float(getattr(state, "_last_step_tvc_gimbal_deg", 0.0))
            alpha_deg[sample_idx] = float(getattr(state, "_last_step_alpha_deg", 0.0))
            beta_deg[sample_idx] = float(getattr(state, "_last_step_beta_deg", 0.0))
            cd[sample_idx] = float(getattr(state, "_last_step_cd", 0.0))
            aero_force_n[sample_idx] = float(getattr(state, "_last_step_aero_force_n", 0.0))
            aero_moment_nm[sample_idx] = float(getattr(state, "_last_step_aero_moment_nm", 0.0))

            if touchdown is not None or (
                self.sim_cfg.terminate_on_earth_impact and self.sim_cfg.touchdown is None
            ):
                impact_radius_km = float(self.sim_cfg.earth_impact_radius_km)
                impact = touchdown is not None or float(_norm3(state.position_eci_km)) <= impact_radius_km
                # EARTH_RADIUS_KM is the WGS-84 equatorial semi-major axis and
                # remains the default selector for the ellipsoidal physical
                # surface. A non-default radius is an explicit spherical
                # impact-radius override, even when WGS-84 geodesy is used for
                # navigation and altitude reporting.
                use_wgs84_surface = (
                    self.sim_cfg.use_wgs84_geodesy and impact_radius_km == EARTH_RADIUS_KM
                )
                if touchdown is None and use_wgs84_surface:
                    _, _, alt_check = _geodetic_state_from_eci(
                        state.position_eci_km,
                        state.t_s,
                        jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
                    )
                    impact = bool(alt_check <= 0.0)
                if impact:
                    terminated_early = True
                    termination_reason = "earth_impact" if touchdown is None else touchdown.termination_reason
                    termination_time = float(state.t_s)
                    n = k + 2
                    return RocketSimResult(
                        time_s=t[:n].copy(),
                        position_eci_km=r[:n, :].copy(),
                        velocity_eci_km_s=v[:n, :].copy(),
                        attitude_quat_bn=q[:n, :].copy(),
                        angular_rate_body_rad_s=w[:n, :].copy(),
                        mass_kg=m[:n].copy(),
                        active_stage_index=stg[:n].copy(),
                        throttle_cmd=thr_cmd[:n].copy(),
                        thrust_n=thrust_n[:n].copy(),
                        altitude_km=alt[:n].copy(),
                        latitude_deg=lat_deg[:n].copy(),
                        longitude_deg=lon_deg[:n].copy(),
                        eccentricity=ecc[:n].copy(),
                        sma_km=sma[:n].copy(),
                        dynamic_pressure_pa=q_dyn[:n].copy(),
                        mach=mach[:n].copy(),
                        wind_body_m_s=wind_body_m_s[:n, :].copy(),
                        tvc_gimbal_deg=tvc_gimbal_deg[:n].copy(),
                        alpha_deg=alpha_deg[:n].copy(),
                        beta_deg=beta_deg[:n].copy(),
                        cd=cd[:n].copy(),
                        aero_force_n=aero_force_n[:n].copy(),
                        aero_moment_nm=aero_moment_nm[:n].copy(),
                        inserted=inserted,
                        insertion_time_s=insertion_time,
                        terminated_early=terminated_early,
                        termination_reason=termination_reason,
                        termination_time_s=termination_time,
                        touchdown=state.gear_touchdown if self.gear_model else (None if touchdown is None else touchdown.evidence),
                        landing_gear=gear_evidence(state) if self.gear_model else None,
                    )

            # insertion criterion: altitude near target and low eccentricity while coasting.
            _, _, alt_ins = _geodetic_state_from_eci(
                state.position_eci_km,
                state.t_s,
                jd_utc_start=self.sim_cfg.atmosphere_env.get("jd_utc_start"),
            )
            alt_compare = float(
                alt_ins if self.sim_cfg.use_wgs84_geodesy else _norm3(state.position_eci_km) - EARTH_RADIUS_KM
            )
            near_alt = abs(alt_compare - self.sim_cfg.target_altitude_km) <= self.sim_cfg.target_altitude_tolerance_km
            _, e_now = _orbital_elements_basic(state.position_eci_km, state.velocity_eci_km_s, EARTH_MU_KM3_S2)
            low_e = e_now <= self.sim_cfg.target_eccentricity_max
            if near_alt and low_e and state.active_stage_index >= len(self._stage_dry):
                insertion_hold_counter += dt
                if insertion_hold_counter >= hold_needed:
                    inserted = True
                    insertion_time = state.t_s
                    # truncate arrays
                    n = k + 2
                    return RocketSimResult(
                        time_s=t[:n].copy(),
                        position_eci_km=r[:n, :].copy(),
                        velocity_eci_km_s=v[:n, :].copy(),
                        attitude_quat_bn=q[:n, :].copy(),
                        angular_rate_body_rad_s=w[:n, :].copy(),
                        mass_kg=m[:n].copy(),
                        active_stage_index=stg[:n].copy(),
                        throttle_cmd=thr_cmd[:n].copy(),
                        thrust_n=thrust_n[:n].copy(),
                        altitude_km=alt[:n].copy(),
                        latitude_deg=lat_deg[:n].copy(),
                        longitude_deg=lon_deg[:n].copy(),
                        eccentricity=ecc[:n].copy(),
                        sma_km=sma[:n].copy(),
                        dynamic_pressure_pa=q_dyn[:n].copy(),
                        mach=mach[:n].copy(),
                        wind_body_m_s=wind_body_m_s[:n, :].copy(),
                        tvc_gimbal_deg=tvc_gimbal_deg[:n].copy(),
                        alpha_deg=alpha_deg[:n].copy(),
                        beta_deg=beta_deg[:n].copy(),
                        cd=cd[:n].copy(),
                        aero_force_n=aero_force_n[:n].copy(),
                        aero_moment_nm=aero_moment_nm[:n].copy(),
                        inserted=inserted,
                        insertion_time_s=insertion_time,
                        terminated_early=terminated_early,
                        termination_reason=termination_reason,
                        termination_time_s=termination_time,
                        touchdown=state.gear_touchdown,
                        landing_gear=gear_evidence(state) if self.gear_model else None,
                    )
            else:
                insertion_hold_counter = 0.0

        return RocketSimResult(
            time_s=t,
            position_eci_km=r,
            velocity_eci_km_s=v,
            attitude_quat_bn=q,
            angular_rate_body_rad_s=w,
            mass_kg=m,
            active_stage_index=stg,
            throttle_cmd=thr_cmd,
            thrust_n=thrust_n,
            altitude_km=alt,
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            eccentricity=ecc,
            sma_km=sma,
            dynamic_pressure_pa=q_dyn,
            mach=mach,
            wind_body_m_s=wind_body_m_s,
            tvc_gimbal_deg=tvc_gimbal_deg,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            cd=cd,
            aero_force_n=aero_force_n,
            aero_moment_nm=aero_moment_nm,
            inserted=inserted,
            insertion_time_s=insertion_time,
            terminated_early=terminated_early,
            termination_reason=termination_reason,
            termination_time_s=termination_time,
            touchdown=state.gear_touchdown,
            landing_gear=gear_evidence(state) if self.gear_model else None,
        )

    def step(self, state: RocketState, command: GuidanceCommand, dt_s: float | None = None) -> RocketState:
        """Public single-step API for external scenario orchestrators."""
        h = float(self.sim_cfg.dt_s if dt_s is None else dt_s)
        throttle = _clamp(float(command.throttle), 0.0, 1.0)
        if self.gear_model is not None:
            if not np.isfinite(h) or h <= 0.0:
                raise ValueError("Rocket landing step duration must be finite and positive.")
            return advance_landing(self, state, command, throttle, h)
        if command.landing_gear_commands:
            raise ValueError("Landing gear commands require configured landing gear.")
        return self._step_once(state=state, cmd=command, throttle=throttle, dt_s=h)

    def resolve_touchdown(
        self, start: RocketState, end: RocketState, command: GuidanceCommand
    ) -> RocketTouchdown | None:
        """Inspect a trial step for contact and return its pre-impact state."""
        cfg = self.sim_cfg
        if cfg.touchdown is None:
            return None
        if self.gear_model is not None:
            terminal = end.gear_terminal
            if terminal is None:
                return None
            return RocketTouchdown(end, terminal["outcome"], {
                **terminal,
                "mass_kg": end.mass_kg,
                "position_eci_km": end.position_eci_km.tolist(),
                "velocity_eci_km_s": end.velocity_eci_km_s.tolist(),
                "attitude_quat_bn": end.attitude_quat_bn.tolist(),
                "angular_rate_body_rad_s": end.angular_rate_body_rad_s.tolist(),
            })
        return locate_touchdown(
            start, end, lambda dt_s: self.step(start, command, dt_s=dt_s),
            limits=cfg.touchdown,
            context=frame_context_from_environment(cfg.atmosphere_env),
            use_wgs84=cfg.use_wgs84_geodesy,
            radius_km=cfg.earth_impact_radius_km,
            upright_axis_body=self.vehicle_cfg.thrust_axis_body,
        )

    def _step_once(
        self, state: RocketState, cmd: GuidanceCommand, throttle: float, dt_s: float,
        *, ground_wrench: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> RocketState:
        s = state.copy()
        stage_i = s.active_stage_index
        thrust_n = 0.0
        dm_prop = 0.0
        mass_start_kg = float(s.mass_kg)
        mass_for_thrust_kg = max(mass_start_kg, 1e-9)
        mass_for_forces_kg = max(mass_start_kg, 1e-9)
        stage_separated = False
        pending_stage_separation = False
        env = {"atmosphere_model": self.sim_cfg.atmosphere_model, **dict(self.sim_cfg.atmosphere_env)}
        if self.sim_cfg.use_wgs84_geodesy:
            env["geodetic_model"] = "wgs84"
        atmos = atmosphere_state_from_model(
            model=str(self.sim_cfg.atmosphere_model).lower(),
            r_eci_km=s.position_eci_km,
            t_s=s.t_s,
            env=env,
        )
        ambient_pressure_pa = float(atmos["pressure_pa"])

        if stage_i < len(self._stage_dry):
            prop_left = float(s.stage_prop_remaining_kg[stage_i])
            if prop_left > 0.0 and throttle > 0.0:
                stage = self.vehicle_cfg.stack.stages[stage_i]
                if str(self.sim_cfg.numeric_backend).strip().lower() == "rust":
                    from sim.rust_vehicle_backend import rocket_propellant_step

                    propellant_step = rocket_propellant_step(
                        propellant_left_kg=prop_left,
                        throttle=throttle,
                        pressure_pa=ambient_pressure_pa,
                        dt_s=dt_s,
                        mass_start_kg=mass_start_kg,
                        sea_level_thrust_n=float(
                            stage.sea_level_thrust_n
                            if stage.sea_level_thrust_n is not None
                            else stage.max_thrust_n
                        ),
                        vacuum_thrust_n=float(
                            stage.vacuum_thrust_n
                            if stage.vacuum_thrust_n is not None
                            else stage.max_thrust_n
                        ),
                        sea_level_isp_s=float(
                            stage.sea_level_isp_s if stage.sea_level_isp_s is not None else stage.isp_s
                        ),
                        vacuum_isp_s=float(
                            stage.vacuum_isp_s if stage.vacuum_isp_s is not None else stage.isp_s
                        ),
                    )
                    thrust_n = float(propellant_step[0])
                    dm_prop = float(propellant_step[2])
                    mass_for_thrust_kg = float(propellant_step[4])
                else:
                    stage_thrust_n, stage_isp_s = _stage_engine_perf(stage, ambient_pressure_pa, numeric_backend=self.sim_cfg.numeric_backend)
                    full_step_thrust_n = float(throttle * stage_thrust_n)
                    mdot = full_step_thrust_n / max(stage_isp_s * G0_M_S2, 1e-9)
                    full_step_dm_prop = mdot * dt_s
                    dm_prop = min(prop_left, full_step_dm_prop)
                    burn_fraction = 0.0 if full_step_dm_prop <= 0.0 else float(dm_prop / full_step_dm_prop)
                    thrust_n = full_step_thrust_n * burn_fraction
                    mass_for_thrust_kg = max(mass_start_kg - 0.5 * dm_prop, 1e-9)
                s.stage_prop_remaining_kg[stage_i] = prop_left - dm_prop
                s.mass_kg = max(0.0, s.mass_kg - dm_prop)

            mass_for_forces_kg = max(s.mass_kg, 1e-9)
            # Defer the dry-mass drop until after this interval's old-stage
            # aero/orbit propagation, so old-stage geometry is not combined
            # with post-separation mass.
            if s.stage_prop_remaining_kg[stage_i] <= 1e-9:
                pending_stage_separation = True

        nominal_thrust_axis_body = _unit(np.asarray(self.vehicle_cfg.thrust_axis_body, dtype=float))
        tvc_target_body = (
            nominal_thrust_axis_body
            if cmd.thrust_vector_body_cmd is None
            else _unit(np.asarray(cmd.thrust_vector_body_cmd, dtype=float))
        )
        s.thrust_vector_body = _step_tvc_vector(
            current_body=s.thrust_vector_body,
            target_body=tvc_target_body,
            nominal_axis_body=nominal_thrust_axis_body,
            dt_s=dt_s,
            sim_cfg=self.sim_cfg,
        )

        mode = str(self.sim_cfg.attitude_mode).strip().lower()
        if mode == "cheater" and cmd.attitude_quat_bn_cmd is not None:
            s.attitude_quat_bn = normalize_quaternion(
                np.asarray(cmd.attitude_quat_bn_cmd, dtype=float)
            )
        c_bn = quaternion_to_dcm_bn(s.attitude_quat_bn)
        thrust_axis_eci = c_bn.T @ s.thrust_vector_body
        accel_thrust_eci_km_s2 = (thrust_n / mass_for_thrust_kg) * thrust_axis_eci / 1e3
        torque_aero_body_nm = np.zeros(3)
        accel_aero_eci_km_s2 = np.zeros(3)
        torque_tvc_body_nm = _cross3(
            np.asarray(self.sim_cfg.tvc_pivot_offset_body_m, dtype=float).reshape(3),
            thrust_n * s.thrust_vector_body,
        )
        last_q_dyn = 0.0
        last_mach = 0.0
        last_wind_body_m_s = np.zeros(3)
        last_tvc_gimbal_deg = _vector_angle_deg(s.thrust_vector_body, nominal_thrust_axis_body)
        last_alpha_deg = 0.0
        last_beta_deg = 0.0
        last_cd = 0.0
        last_aero_force_n = 0.0
        last_aero_moment_nm = 0.0
        v_rel_eci_m_s, wind_eci_m_s = rocket_air_relative_state_eci_m_s(
            position_eci_km=s.position_eci_km,
            velocity_eci_km_s=s.velocity_eci_km_s,
            t_s=s.t_s,
            sim_cfg=self.sim_cfg,
            state=s,
        )
        v_rel_body_m_s = c_bn @ v_rel_eci_m_s
        if self.sim_cfg.enable_drag or self.sim_cfg.aero.enabled:
            speed_m_s = float(_norm3(v_rel_eci_m_s))
            last_q_dyn = 0.5 * float(atmos["density_kg_m3"]) * speed_m_s * speed_m_s
            sound_speed_m_s = float(max(float(atmos.get("sound_speed_m_s", 0.0)), 1e-9))
            last_mach = speed_m_s / sound_speed_m_s
            last_wind_body_m_s = c_bn @ wind_eci_m_s
        if self.sim_cfg.aero.enabled:
            aero_state = compute_aero_state(
                rho_kg_m3=float(atmos["density_kg_m3"]),
                pressure_pa=ambient_pressure_pa,
                temperature_k=float(atmos["temperature_k"]),
                sound_speed_m_s=float(atmos["sound_speed_m_s"]),
                v_rel_body_m_s=v_rel_body_m_s,
                alpha_limit_deg=float(self.sim_cfg.aero.alpha_limit_deg),
                beta_limit_deg=float(self.sim_cfg.aero.beta_limit_deg),
                numeric_backend=self.sim_cfg.numeric_backend,
            )
            aero_cfg = self._resolve_aero_config_for_stage(stage_i=stage_i)
            loads = compute_aero_loads(
                v_rel_body_m_s=v_rel_body_m_s,
                atmos=aero_state,
                cfg=aero_cfg,
                numeric_backend=self.sim_cfg.numeric_backend,
            )
            f_aero_eci_n = c_bn.T @ loads.force_body_n
            accel_aero_eci_km_s2 = f_aero_eci_n / max(mass_for_forces_kg, 1e-9) / 1e3
            torque_aero_body_nm = loads.moment_body_nm
            last_q_dyn = float(loads.state.dynamic_pressure_pa)
            last_mach = float(loads.state.mach)
            last_wind_body_m_s = c_bn @ wind_eci_m_s
            last_alpha_deg = float(np.rad2deg(loads.state.alpha_rad))
            last_beta_deg = float(np.rad2deg(loads.state.beta_rad))
            last_cd = float(loads.drag_coefficient)
            last_aero_force_n = float(_norm3(loads.force_body_n))
            last_aero_moment_nm = float(_norm3(loads.moment_body_nm))

        x_orbit = np.empty(6, dtype=float)
        x_orbit[:3] = s.position_eci_km
        x_orbit[3:] = s.velocity_eci_km_s
        ctx = OrbitContext(
            mu_km3_s2=EARTH_MU_KM3_S2,
            mass_kg=mass_for_forces_kg,
            area_m2=self._area_ref_m2,
            cd=self.sim_cfg.cd,
            cr=self.sim_cfg.cr,
        )
        command_accel = accel_thrust_eci_km_s2 + accel_aero_eci_km_s2
        if ground_wrench is not None:
            command_accel += ground_wrench[0] / (1000.0 * mass_for_forces_kg)
        x_next = self._propagator.propagate(
            x_eci=x_orbit,
            dt_s=dt_s,
            t_s=s.t_s,
            command_accel_eci_km_s2=command_accel,
            env=env,
            ctx=ctx,
        )

        if mode == "cheater":
            if cmd.attitude_quat_bn_cmd is not None:
                qn = normalize_quaternion(np.asarray(cmd.attitude_quat_bn_cmd, dtype=float))
            else:
                qn = s.attitude_quat_bn.copy()
            wn = np.zeros(3, dtype=float)
        else:
            torque_cmd = (
                np.zeros(3) if cmd.torque_body_nm_cmd is None else np.array(cmd.torque_body_nm_cmd, dtype=float)
            )
            torque_cmd = torque_cmd + torque_aero_body_nm + torque_tvc_body_nm
            torque_with_contact = torque_cmd if ground_wrench is None else torque_cmd + ground_wrench[1]
            att_h = max(min(self.sim_cfg.attitude_substep_s, dt_s), 1e-4)
            rem = dt_s
            qn = s.attitude_quat_bn.copy()
            wn = s.angular_rate_body_rad_s.copy()
            while rem > 1e-12:
                h = min(att_h, rem)
                qn, wn = propagate_attitude_exponential_map(
                    quat_bn=qn,
                    omega_body_rad_s=wn,
                    inertia_kg_m2=self.sim_cfg.inertia_kg_m2,
                    torque_body_nm=torque_with_contact,
                    dt_s=h,
                    numeric_backend=self.sim_cfg.numeric_backend,
                )
                rem -= h

        s.position_eci_km = x_next[:3]
        s.velocity_eci_km_s = x_next[3:]
        s.attitude_quat_bn = qn
        s.angular_rate_body_rad_s = wn
        s.t_s = s.t_s + dt_s
        if pending_stage_separation and stage_i < len(self._stage_dry):
            s.mass_kg = max(0.0, s.mass_kg - self._stage_dry[stage_i])
            s.active_stage_index += 1
            stage_separated = True
        # attach last-step telemetry attribute for logging convenience.
        s._last_step_thrust_n = thrust_n
        s._last_step_stage_sep = stage_separated
        s._last_step_force_mass_kg = mass_for_forces_kg
        s._last_step_thrust_axis_eci = thrust_axis_eci.copy()
        s._last_step_thrust_accel_eci_km_s2 = accel_thrust_eci_km_s2.copy()
        s._last_step_q_dyn_pa = last_q_dyn
        s._last_step_mach = last_mach
        s._last_step_wind_body_m_s = last_wind_body_m_s.copy()
        s._last_step_tvc_gimbal_deg = last_tvc_gimbal_deg
        s._last_step_alpha_deg = last_alpha_deg
        s._last_step_beta_deg = last_beta_deg
        s._last_step_cd = last_cd
        s._last_step_aero_force_n = last_aero_force_n
        s._last_step_aero_moment_nm = last_aero_moment_nm
        s._last_step_torque_body_nm = (
            np.zeros(3, dtype=float) if mode == "cheater" else np.array(torque_cmd, dtype=float).reshape(3)
        )
        return s
