from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sim.dynamics.orbit.atmosphere import atmosphere_state_from_model
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.rocket.models import GuidanceCommand, RocketGuidanceLaw, RocketSimConfig, RocketState, RocketVehicleConfig
from sim.rocket.navigation import (
    _clip_scalar,
    _cross3,
    _geodetic_state_from_eci,
    _norm3,
    atmosphere_relative_velocity_rocket_eci_m_s,
    build_rocket_nav_state,
)
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(_norm3(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def _quat_from_body_x_and_hint(x_axis_eci: np.ndarray, z_hint_eci: np.ndarray) -> np.ndarray:
    x_hat = _unit(np.asarray(x_axis_eci, dtype=float))
    if _norm3(x_hat) <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    z_hint = _unit(np.asarray(z_hint_eci, dtype=float))
    y_hat = _unit(_cross3(z_hint, x_hat))
    if _norm3(y_hat) <= 0.0:
        # fallback if collinear
        y_hat = _unit(_cross3(np.array([0.0, 0.0, 1.0]), x_hat))
        if _norm3(y_hat) <= 0.0:
            y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = _unit(_cross3(x_hat, y_hat))
    c_bn = np.vstack((x_hat, y_hat, z_hat))  # body rows in inertial components
    return dcm_to_quaternion_bn(c_bn)


def _orbital_elements_basic(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float]:
    r = float(_norm3(r_km))
    v2 = float(np.dot(v_km_s, v_km_s))
    if r <= 0.0:
        return np.inf, np.inf
    eps = 0.5 * v2 - mu_km3_s2 / r
    a = np.inf if abs(eps) < 1e-14 else float(-mu_km3_s2 / (2.0 * eps))
    h = _cross3(r_km, v_km_s)
    e_vec = _cross3(v_km_s, h) / mu_km3_s2 - r_km / r
    e = float(_norm3(e_vec))
    return a, e


def _apo_peri_alt_km(
    r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = EARTH_MU_KM3_S2
) -> tuple[float, float, float, float]:
    a_km, e = _orbital_elements_basic(r_km, v_km_s, mu_km3_s2)
    if not np.isfinite(a_km) or a_km <= 0.0:
        return np.inf, -np.inf, a_km, e
    ra = float(a_km * (1.0 + e))
    rp = float(a_km * (1.0 - e))
    return ra - EARTH_RADIUS_KM, rp - EARTH_RADIUS_KM, a_km, e


@dataclass(frozen=True)
class OpenLoopPitchProgramGuidance(RocketGuidanceLaw):
    """Simple launch guidance: hold vertical, then pitch over and follow velocity direction."""

    vertical_hold_s: float = 10.0
    pitch_start_s: float = 10.0
    pitch_end_s: float = 180.0
    pitch_final_deg: float = 70.0
    max_throttle: float = 1.0
    min_throttle: float = 0.0

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        t = state.t_s
        r_hat = _unit(state.position_eci_km)
        v_hat = _unit(state.velocity_eci_km_s)
        east_hat = _unit(_cross3(np.array([0.0, 0.0, 1.0]), r_hat))
        if _norm3(east_hat) <= 0.0:
            east_hat = np.array([0.0, 1.0, 0.0])

        if t <= self.vertical_hold_s:
            x_cmd = r_hat
        elif t <= self.pitch_end_s:
            alpha = float(
                _clip_scalar((t - self.pitch_start_s) / max(self.pitch_end_s - self.pitch_start_s, 1e-9), 0.0, 1.0)
            )
            pitch_rad = np.deg2rad(alpha * self.pitch_final_deg)
            x_cmd = _unit(np.cos(pitch_rad) * r_hat + np.sin(pitch_rad) * east_hat)
        else:
            if _norm3(v_hat) > 0.0:
                x_cmd = _unit(v_hat)
            else:
                x_cmd = r_hat

        q_cmd = _quat_from_body_x_and_hint(x_cmd, z_hint_eci=r_hat)
        thr = float(_clip_scalar(self.max_throttle, self.min_throttle, self.max_throttle))
        return GuidanceCommand(throttle=thr, attitude_quat_bn_cmd=q_cmd, torque_body_nm_cmd=np.zeros(3))


@dataclass
class ClosedLoopInsertionGuidance(RocketGuidanceLaw):
    """Closed-loop ascent guidance with simple phase logic for insertion targeting.

    Phases:
    - ascent: gravity-turn style pitch with apogee growth target
    - coast_to_apogee: hold near-prograde while waiting for apogee
    - circularize: near-apogee prograde burn to raise perigee / reduce eccentricity
    - complete: cutoff
    """

    max_throttle: float = 1.0
    min_throttle: float = 0.0
    turn_start_alt_km: float = 3.0
    turn_end_alt_km: float = 70.0
    apoapsis_switch_margin_km: float = 5.0
    circularize_window_alt_km: float = 40.0
    circularize_dv_tol_km_s: float = 0.005
    periapsis_target_margin_km: float = 20.0
    radial_damp_gain: float = 4.0
    tangential_gain: float = 1.0
    desired_inc_deg: float | None = None
    desired_raan_deg: float | None = None
    plane_alignment_gain: float = 0.5
    speed_cap_frac_of_escape: float = 0.88
    speed_cap_soften_gain: float = 1.0
    speed_cap_min_throttle_frac: float = 0.15
    hyperbolic_guard_enabled: bool = True
    hyperbolic_guard_margin_km2_s2: float = 0.01
    hyperbolic_guard_lookahead_s: float = 2.0
    energy_ceiling_enabled: bool = True
    energy_ceiling_start_alt_km: float = 100.0
    energy_ceiling_margin_km2_s2: float = 0.5

    _phase: str = "ascent"
    _last_radial_speed_km_s: float = 0.0

    def _horizontal_azimuth_dir(self, r_hat: np.ndarray, azimuth_deg: float) -> np.ndarray:
        k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
        east_hat = _unit(_cross3(k_hat, r_hat))
        if _norm3(east_hat) <= 0.0:
            east_hat = np.array([0.0, 1.0, 0.0], dtype=float)
        north_hat = _unit(_cross3(r_hat, east_hat))
        az = np.deg2rad(float(azimuth_deg))
        return _unit(np.cos(az) * north_hat + np.sin(az) * east_hat)

    def _cmd_from_axis(self, thrust_axis_eci: np.ndarray, throttle: float, r_hat: np.ndarray) -> GuidanceCommand:
        q_cmd = _quat_from_body_x_and_hint(_unit(thrust_axis_eci), z_hint_eci=r_hat)
        thr = float(_clip_scalar(throttle, self.min_throttle, self.max_throttle))
        return GuidanceCommand(throttle=thr, attitude_quat_bn_cmd=q_cmd, torque_body_nm_cmd=np.zeros(3))

    def _apply_speed_cap(self, throttle: float, v_mag_km_s: float, r_norm_km: float) -> float:
        if self.max_throttle <= 0.0:
            return 0.0
        frac = float(_clip_scalar(self.speed_cap_frac_of_escape, 0.5, 0.999))
        v_esc = float(np.sqrt(2.0 * EARTH_MU_KM3_S2 / max(r_norm_km, 1e-9)))
        v_cap = frac * v_esc
        if v_mag_km_s <= v_cap:
            return float(_clip_scalar(throttle, self.min_throttle, self.max_throttle))
        headroom = max((1.0 - frac) * v_esc, 1e-6)
        excess = max(v_mag_km_s - v_cap, 0.0)
        soften = float(max(self.speed_cap_soften_gain, 0.0))
        scale = float(_clip_scalar(1.0 - soften * (excess / headroom), 0.0, 1.0))
        floor = float(_clip_scalar(self.speed_cap_min_throttle_frac, 0.0, 1.0))
        capped = self.max_throttle * max(scale, floor)
        return float(_clip_scalar(min(throttle, capped), self.min_throttle, self.max_throttle))

    def _estimate_max_accel_km_s2(self, state: RocketState, vehicle_cfg: RocketVehicleConfig) -> float:
        i = int(state.active_stage_index)
        if i < 0 or i >= len(vehicle_cfg.stack.stages):
            return 0.0
        stage = vehicle_cfg.stack.stages[i]
        if float(state.mass_kg) <= 0.0:
            return 0.0
        return float(max(stage.max_thrust_n, 0.0) / max(float(state.mass_kg), 1e-9) / 1e3)

    def _apply_hyperbolic_guard(
        self,
        throttle: float,
        thrust_dir_eci: np.ndarray,
        state: RocketState,
        vehicle_cfg: RocketVehicleConfig,
        sim_cfg: RocketSimConfig,
    ) -> float:
        thr = float(_clip_scalar(throttle, self.min_throttle, self.max_throttle))
        if (not self.hyperbolic_guard_enabled) or thr <= 0.0:
            return thr
        r = np.array(state.position_eci_km, dtype=float).reshape(3)
        v = np.array(state.velocity_eci_km_s, dtype=float).reshape(3)
        r_norm = float(_norm3(r))
        if r_norm <= 0.0:
            return 0.0
        _, _, alt_geo_km = _geodetic_state_from_eci(
            state.position_eci_km,
            state.t_s,
            jd_utc_start=sim_cfg.atmosphere_env.get("jd_utc_start"),
        )
        alt_km = float(alt_geo_km if sim_cfg.use_wgs84_geodesy else r_norm - EARTH_RADIUS_KM)
        u = _unit(np.asarray(thrust_dir_eci, dtype=float).reshape(3))
        if _norm3(u) <= 0.0:
            return 0.0
        a_max = self._estimate_max_accel_km_s2(state, vehicle_cfg)
        if a_max <= 0.0:
            return 0.0
        dt = float(max(self.hyperbolic_guard_lookahead_s, 1e-6))
        alpha_per_throttle = a_max * dt
        v2 = float(np.dot(v, v))
        b = float(np.dot(v, u))
        c_hyp = 0.5 * v2 - float(EARTH_MU_KM3_S2 / r_norm) + float(max(self.hyperbolic_guard_margin_km2_s2, 0.0))

        # Ensure 0.5*|v + alpha*u|^2 - mu/r + margin <= 0
        # => alpha^2 + 2*b*alpha + 2*c <= 0
        def _thr_limit_from_c(c_val: float) -> float:
            disc = b * b - 2.0 * c_val
            if disc <= 0.0:
                return 0.0
            alpha_hi = -b + float(np.sqrt(disc))
            if alpha_hi <= 0.0:
                return 0.0
            return float(_clip_scalar(alpha_hi / max(alpha_per_throttle, 1e-9), 0.0, self.max_throttle))

        thr_limit = _thr_limit_from_c(c_hyp)

        if self.energy_ceiling_enabled and alt_km >= float(max(self.energy_ceiling_start_alt_km, 0.0)):
            r_tgt = float(EARTH_RADIUS_KM + sim_cfg.target_altitude_km)
            eps_tgt = float(-EARTH_MU_KM3_S2 / (2.0 * max(r_tgt, 1e-6)))
            eps_lim = float(eps_tgt + max(self.energy_ceiling_margin_km2_s2, 0.0))
            c_energy = 0.5 * v2 - float(EARTH_MU_KM3_S2 / r_norm) - eps_lim
            thr_limit = min(thr_limit, _thr_limit_from_c(c_energy))

        return float(_clip_scalar(min(thr, thr_limit), self.min_throttle, self.max_throttle))

    def _desired_plane_normal(self) -> np.ndarray | None:
        if self.desired_inc_deg is None or self.desired_raan_deg is None:
            return None
        inc = np.deg2rad(float(self.desired_inc_deg))
        raan = np.deg2rad(float(self.desired_raan_deg))
        h_hat = np.array(
            [
                np.sin(inc) * np.sin(raan),
                -np.sin(inc) * np.cos(raan),
                np.cos(inc),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(h_hat)) or _norm3(h_hat) <= 0.0:
            return None
        return _unit(h_hat)

    def _guided_tangential_dir(self, r_hat: np.ndarray, v: np.ndarray, fallback_t_hat: np.ndarray) -> np.ndarray:
        h_des = self._desired_plane_normal()
        if h_des is None:
            return _unit(fallback_t_hat)
        t_des = _unit(_cross3(h_des, r_hat))
        if _norm3(t_des) <= 0.0:
            return _unit(fallback_t_hat)
        if float(np.dot(t_des, v)) < 0.0:
            t_des = -t_des
        g = float(_clip_scalar(self.plane_alignment_gain, 0.0, 1.0))
        return _unit((1.0 - g) * _unit(fallback_t_hat) + g * t_des)

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        r = np.array(state.position_eci_km, dtype=float).reshape(3)
        v = np.array(state.velocity_eci_km_s, dtype=float).reshape(3)
        r_norm = float(_norm3(r))
        if r_norm <= 0.0:
            return GuidanceCommand(throttle=0.0, attitude_quat_bn_cmd=None, torque_body_nm_cmd=np.zeros(3))
        r_hat = _unit(r)
        v_mag = float(_norm3(v))
        v_hat = _unit(v) if v_mag > 1e-9 else r_hat
        h_hat = _unit(_cross3(r, v))
        t_hat = _unit(_cross3(h_hat, r_hat))
        if _norm3(t_hat) <= 0.0:
            t_hat = self._horizontal_azimuth_dir(r_hat, sim_cfg.launch_azimuth_deg)
        t_guided = self._guided_tangential_dir(r_hat, v, t_hat)

        nav = build_rocket_nav_state(state, sim_cfg, vehicle_cfg, throttle_cmd=self.max_throttle)
        alt_km = nav.altitude_km
        target_alt_km = float(sim_cfg.target_altitude_km)
        target_r_km = EARTH_RADIUS_KM + target_alt_km
        ra_alt = nav.apoapsis_alt_km
        rp_alt = nav.periapsis_alt_km
        ecc = nav.eccentricity
        vr = nav.vertical_speed_km_s
        vt = float(np.dot(v, t_guided))

        # Phase transitions.
        if self._phase == "ascent" and ra_alt >= (target_alt_km - float(max(self.apoapsis_switch_margin_km, 0.0))):
            self._phase = "coast_to_apogee"
        if self._phase == "coast_to_apogee":
            near_target_radius = abs(r_norm - target_r_km) <= float(max(self.circularize_window_alt_km, 0.0))
            crossed_apo = self._last_radial_speed_km_s > 0.0 and vr <= 0.0
            if near_target_radius or crossed_apo:
                self._phase = "circularize"
        if self._phase == "circularize":
            v_circ_here = float(np.sqrt(EARTH_MU_KM3_S2 / max(r_norm, 1e-9)))
            dv_need = max(v_circ_here - vt, 0.0)
            peri_ok = rp_alt >= (target_alt_km - float(max(self.periapsis_target_margin_km, 0.0)))
            ecc_ok = ecc <= float(max(sim_cfg.target_eccentricity_max, 0.0))
            if peri_ok and ecc_ok and dv_need <= float(max(self.circularize_dv_tol_km_s, 0.0)):
                self._phase = "complete"
        self._last_radial_speed_km_s = vr

        if self._phase == "complete":
            return self._cmd_from_axis(v_hat, throttle=0.0, r_hat=r_hat)

        if self._phase == "coast_to_apogee":
            # Coast to avoid overshooting energy while preserving near-prograde attitude.
            return self._cmd_from_axis(v_hat, throttle=0.0, r_hat=r_hat)

        if self._phase == "circularize":
            v_circ_here = float(np.sqrt(EARTH_MU_KM3_S2 / max(r_norm, 1e-9)))
            dv_need = max(v_circ_here - vt, 0.0)
            throttle = float(_clip_scalar(dv_need / 0.15, 0.0, self.max_throttle))
            throttle = self._apply_speed_cap(throttle, v_mag, r_norm)
            throttle = self._apply_hyperbolic_guard(throttle, t_guided, state, vehicle_cfg, sim_cfg)
            return self._cmd_from_axis(t_guided, throttle=throttle, r_hat=r_hat)

        # Ascent phase: blend radial -> horizontal and damp radial overspeed.
        turn_span = max(float(self.turn_end_alt_km - self.turn_start_alt_km), 1e-6)
        turn_alpha = float(_clip_scalar((alt_km - float(self.turn_start_alt_km)) / turn_span, 0.0, 1.0))
        horiz_dir = _unit(
            (1.0 - float(_clip_scalar(self.plane_alignment_gain, 0.0, 1.0)))
            * self._horizontal_azimuth_dir(r_hat, sim_cfg.launch_azimuth_deg)
            + float(_clip_scalar(self.plane_alignment_gain, 0.0, 1.0)) * t_guided
        )
        cmd_dir = _unit((1.0 - turn_alpha) * r_hat + turn_alpha * horiz_dir)
        if _norm3(t_guided) > 0.0:
            cmd_dir = _unit(self.tangential_gain * cmd_dir + max(-vr, 0.0) * self.radial_damp_gain * t_guided)
        throttle_ascent = self._apply_speed_cap(self.max_throttle, v_mag, r_norm)
        throttle_ascent = self._apply_hyperbolic_guard(throttle_ascent, cmd_dir, state, vehicle_cfg, sim_cfg)
        return self._cmd_from_axis(cmd_dir, throttle=throttle_ascent, r_hat=r_hat)


@dataclass(frozen=True)
class HoldAttitudeGuidance(RocketGuidanceLaw):
    throttle: float = 1.0

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        return GuidanceCommand(
            throttle=float(_clip_scalar(self.throttle, 0.0, 1.0)), attitude_quat_bn_cmd=None, torque_body_nm_cmd=np.zeros(3)
        )


@dataclass(frozen=True)
class TVCSteeringGuidance(RocketGuidanceLaw):
    """Wrap a guidance law and convert its desired thrust axis into a body-frame TVC command."""

    base_guidance: RocketGuidanceLaw
    pass_through_attitude: bool = False

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        cmd = self.base_guidance.command(state, sim_cfg, vehicle_cfg)
        if cmd.attitude_quat_bn_cmd is None:
            return cmd
        thrust_axis_body = _unit(np.asarray(vehicle_cfg.thrust_axis_body, dtype=float))
        c_cmd_bn = quaternion_to_dcm_bn(np.asarray(cmd.attitude_quat_bn_cmd, dtype=float))
        thrust_axis_eci_cmd = c_cmd_bn.T @ thrust_axis_body
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        thrust_vector_body_cmd = c_bn @ thrust_axis_eci_cmd
        return replace(
            cmd,
            attitude_quat_bn_cmd=cmd.attitude_quat_bn_cmd if self.pass_through_attitude else None,
            thrust_vector_body_cmd=thrust_vector_body_cmd,
        )


@dataclass(frozen=True)
class MaxQThrottleLimiterGuidance(RocketGuidanceLaw):
    """Wrap a base guidance law and limit throttle when dynamic pressure exceeds max_q."""

    base_guidance: RocketGuidanceLaw
    max_q_pa: float = 45_000.0
    min_throttle: float = 0.0

    def _estimate_dynamic_pressure_pa(self, state: RocketState, sim_cfg: RocketSimConfig) -> float:
        env = {"atmosphere_model": sim_cfg.atmosphere_model, **dict(sim_cfg.atmosphere_env)}
        if sim_cfg.use_wgs84_geodesy:
            env["geodetic_model"] = "wgs84"
        atmos = atmosphere_state_from_model(
            model=str(sim_cfg.atmosphere_model).lower(),
            r_eci_km=state.position_eci_km,
            t_s=state.t_s,
            env=env,
        )
        rho = float(max(atmos["density_kg_m3"], 0.0))
        c_bn = quaternion_to_dcm_bn(state.attitude_quat_bn)
        v_rel_eci_m_s = atmosphere_relative_velocity_rocket_eci_m_s(
            position_eci_km=state.position_eci_km,
            velocity_eci_km_s=state.velocity_eci_km_s,
            t_s=state.t_s,
            sim_cfg=sim_cfg,
            state=state,
        )
        v_rel_body_m_s = c_bn @ v_rel_eci_m_s
        speed = float(_norm3(v_rel_body_m_s))
        return 0.5 * rho * speed * speed

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        cmd = self.base_guidance.command(state, sim_cfg, vehicle_cfg)
        thr_cmd = float(_clip_scalar(cmd.throttle, 0.0, 1.0))
        if self.max_q_pa <= 0.0 or thr_cmd <= 0.0:
            return cmd

        q_now = self._estimate_dynamic_pressure_pa(state=state, sim_cfg=sim_cfg)
        if q_now <= self.max_q_pa:
            return cmd

        scale = float(_clip_scalar(self.max_q_pa / max(q_now, 1e-9), 0.0, 1.0))
        thr_limited = float(_clip_scalar(thr_cmd * scale, self.min_throttle, thr_cmd))
        return replace(cmd, throttle=thr_limited)


@dataclass(frozen=True)
class OrbitInsertionCutoffGuidance(RocketGuidanceLaw):
    """Wrap a base guidance law and cut throttle to avoid overshooting to escape/hyperbolic energy."""

    base_guidance: RocketGuidanceLaw
    min_cutoff_alt_km: float = 80.0
    min_periapsis_alt_km: float = 120.0
    apoapsis_margin_km: float = 5.0
    energy_margin_km2_s2: float = 0.0
    ecc_relax_factor: float = 2.0
    hard_escape_cutoff: bool = True
    near_escape_speed_margin_frac: float = 0.03

    def _should_cutoff(self, state: RocketState, sim_cfg: RocketSimConfig) -> tuple[bool, str]:
        r = np.array(state.position_eci_km, dtype=float).reshape(3)
        v = np.array(state.velocity_eci_km_s, dtype=float).reshape(3)
        r_norm = float(_norm3(r))
        if r_norm <= 0.0:
            return False, "invalid_radius"
        _, _, alt_geo_km = _geodetic_state_from_eci(
            state.position_eci_km,
            state.t_s,
            jd_utc_start=sim_cfg.atmosphere_env.get("jd_utc_start"),
        )
        alt_km = float(alt_geo_km if sim_cfg.use_wgs84_geodesy else r_norm - EARTH_RADIUS_KM)
        v_mag = float(_norm3(v))
        if alt_km < float(max(self.min_cutoff_alt_km, 0.0)):
            return False, "below_cutoff_altitude"

        mu = float(EARTH_MU_KM3_S2)
        v2 = float(np.dot(v, v))
        eps = 0.5 * v2 - mu / r_norm
        if self.hard_escape_cutoff and eps >= 0.0:
            return True, "escape_energy"

        a_km, e = _orbital_elements_basic(r, v, mu)
        if not np.isfinite(a_km) or a_km <= 0.0:
            return False, "invalid_elements"
        rp_km = float(a_km * (1.0 - e))
        ra_km = float(a_km * (1.0 + e))
        target_r_km = float(EARTH_RADIUS_KM + sim_cfg.target_altitude_km)
        periapsis_ok = rp_km >= float(EARTH_RADIUS_KM + max(self.min_periapsis_alt_km, 0.0))

        # Pre-escape protection should only shut down once perigee is also safe.
        v_esc = float(np.sqrt(2.0 * mu / r_norm))
        margin = float(_clip_scalar(self.near_escape_speed_margin_frac, 0.0, 0.5))
        if periapsis_ok and v_mag >= (1.0 - margin) * v_esc:
            return True, "near_escape_speed"
        if not periapsis_ok:
            return False, "perigee_too_low"
        if ra_km >= (target_r_km - float(max(self.apoapsis_margin_km, 0.0))):
            return True, "apogee_target_reached"

        eps_target = float(-mu / (2.0 * max(target_r_km, 1e-6)))
        e_lim = float(max(sim_cfg.target_eccentricity_max, 0.0) * max(self.ecc_relax_factor, 1.0))
        if eps >= (eps_target - float(max(self.energy_margin_km2_s2, 0.0))) and e <= e_lim:
            return True, "energy_target_reached"
        return False, "continue_burn"

    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand:
        cmd = self.base_guidance.command(state, sim_cfg, vehicle_cfg)
        thr_cmd = float(_clip_scalar(cmd.throttle, 0.0, 1.0))
        if thr_cmd <= 0.0:
            return cmd
        cut, _ = self._should_cutoff(state=state, sim_cfg=sim_cfg)
        if not cut:
            return cmd
        return replace(cmd, throttle=0.0)
