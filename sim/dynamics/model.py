from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import sim.aero.core as aero_core
from sim.acceleration.settings import acceleration_enabled_from_mode
from sim.core.interfaces import DynamicsModel
from sim.core.models import Command, StateTruth
from sim.dynamics.attitude.disturbances import DisturbanceTorqueModel
from sim.dynamics.attitude.rigid_body import guard_attitude_state_output, propagate_attitude_exponential_map
from sim.dynamics.coupled_satellite import (
    CoupledDerivative,
    CoupledIntegratorConfig,
    CoupledSatelliteIntegrator,
    CoupledSatelliteState,
)
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.atmosphere import density_from_model
from sim.dynamics.orbit.eclipse import resolve_srp_geometry, srp_shadow_factor
from sim.dynamics.orbit.environment import EARTH_ROT_RATE_RAD_S
from sim.dynamics.orbit.epoch import resolve_time_dependent_env
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.dynamics.spacecraft_geometry import GeometryAreaProfile, RectangularPrismGeometry
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.quaternion import quaternion_to_dcm_bn


def _owned_default_orbit_propagator() -> OrbitPropagator:
    propagator = OrbitPropagator(integrator="rk4")
    propagator._pending_orbital_attitude_default_configuration = True
    return propagator


@dataclass(frozen=True)
class OrbitalAttitudeDynamics(DynamicsModel):
    mu_km3_s2: float
    inertia_kg_m2: np.ndarray
    disturbance_model: DisturbanceTorqueModel | None = None
    area_m2: float = 1.0
    cd: float = 2.2
    cr: float = 1.2
    drag_area_m2: float | None = None
    lift_area_m2: float | None = None
    lift_coefficient: float = 0.0
    lift_axis_body: np.ndarray | None = None
    srp_area_m2: float | None = None
    use_rectangular_prism_for_aero_srp: bool = False
    rectangular_prism_dims_m: tuple[float, float, float] | None = None
    geometry_area_profile: GeometryAreaProfile | None = None
    orbit_substep_s: float | None = None
    attitude_substep_s: float | None = None
    propagate_attitude: bool = True
    orbit_propagator: OrbitPropagator = field(default_factory=_owned_default_orbit_propagator)
    acceleration_mode: str = "off"
    resource_model: object | None = None
    attitude_numeric_backend: str = "rust"
    _acceleration_enabled: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        backend = normalize_numeric_backend(self.attitude_numeric_backend, error_message="attitude_numeric_backend must be python or rust")
        object.__setattr__(self, "attitude_numeric_backend", backend)
        if self.geometry_area_profile is not None and self.use_rectangular_prism_for_aero_srp:
            raise ValueError("Use either geometry_area_profile or rectangular prism aero/SRP mode, not both.")
        if self.use_rectangular_prism_for_aero_srp:
            if self.rectangular_prism_dims_m is None:
                raise ValueError(
                    "rectangular_prism_dims_m must be provided when use_rectangular_prism_for_aero_srp=True."
                )
            if self.disturbance_model is None:
                raise ValueError(
                    "Rectangular prism aero/SRP mode requires coupled orbit+attitude disturbance simulation "
                    "(disturbance_model must be set)."
                )
        acceleration_enabled = acceleration_enabled_from_mode(self.acceleration_mode)
        object.__setattr__(self, "_acceleration_enabled", bool(acceleration_enabled))
        if bool(getattr(self.orbit_propagator, "_pending_orbital_attitude_default_configuration", False)):
            self.orbit_propagator.acceleration_mode = self.acceleration_mode
            delattr(self.orbit_propagator, "_pending_orbital_attitude_default_configuration")

    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        try:
            dt_s = float(dt_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("dynamics timestep must be finite and positive") from exc
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dynamics timestep must be finite and positive")
        if self.resource_model is None:
            return self._step_mechanical(state, command, env, dt_s)
        if self.orbit_propagator.state_frame != "eci":
            raise ValueError("spacecraft resources currently require Earth-centered ECI dynamics")
        count = max(1, int(np.ceil(dt_s / self.resource_model.max_step_s)))
        h = dt_s / count
        current = state
        subcommand = Command(
            command.thrust_eci_km_s2, command.torque_body_nm,
            {**command.mode_flags, "delta_mass_kg": float(command.mode_flags.get("delta_mass_kg", 0.0)) / count},
        )
        for _ in range(count):
            following = self._step_mechanical(current, subcommand, env, h)
            following.resource_state = self.resource_model.advance(current, following, env, h)
            current = following
        return current

    def _step_mechanical(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        native_context = self._native_mechanical_context(command, env) if dt_s > 0.0 else None
        if native_context is not None:
            from sim.dynamics.attitude.rigid_body import _add_guardrail_counts

            raw_state = np.concatenate((state.position_eci_km, state.velocity_eci_km_s,
                                        state.attitude_quat_bn, state.angular_rate_body_rad_s))
            acceleration = np.asarray(command.thrust_eci_km_s2, dtype=float)
            if command.mode_flags.get("physical_force_eci_n") is not None and command.mode_flags.get("physical_force_body_n") is not None:
                # The ordinary owner gives an explicitly provided zero wrench
                # precedence over commanded acceleration.
                acceleration = np.zeros(3, dtype=float)
            values, counts = native_context.step(
                raw_state.tolist(), float(dt_s), self.orbit_substep_s, self.attitude_substep_s,
                acceleration.tolist(),
                np.asarray(command.torque_body_nm, dtype=float).tolist(),
            )
            _add_guardrail_counts(counts)
            vector = np.asarray(values, dtype=float)
            self.orbit_propagator.last_numeric_path = "rust_native_mechanical"
            min_mass_kg = float(command.mode_flags.get("min_mass_kg", 0.0))
            if not np.isfinite(min_mass_kg):
                min_mass_kg = 0.0
            min_mass_kg = max(min_mass_kg, 0.0)
            mass_next = max(min_mass_kg, state.mass_kg - float(command.mode_flags.get("delta_mass_kg", 0.0)))
            return StateTruth(vector[:3].copy(), vector[3:6].copy(), vector[6:10].copy(),
                              vector[10:13].copy(), mass_next, state.t_s + dt_s)
        propagate_attitude = self.propagate_attitude and not bool(env.get("attitude_disabled", False))
        force_eci_n = np.asarray(command.mode_flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float)
        force_body_n = np.asarray(command.mode_flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float)
        mass_flow_kg_s = float(command.mode_flags.get("mass_flow_kg_s", 0.0) or 0.0)
        if (
            propagate_attitude
            and self.orbit_propagator.state_frame == "eci"
            and (np.any(force_eci_n) or np.any(force_body_n) or mass_flow_kg_s > 0.0)
        ):
            return self._step_coupled_v2(state, command, env, dt_s)
        env_local = dict(env)
        q_force_evaluation = np.asarray(state.attitude_quat_bn, dtype=float)
        if propagate_attitude and dt_s > 0.0:
            q_force_evaluation, _ = propagate_attitude_exponential_map(
                quat_bn=q_force_evaluation,
                omega_body_rad_s=np.asarray(state.angular_rate_body_rad_s, dtype=float),
                inertia_kg_m2=self.inertia_kg_m2,
                torque_body_nm=np.asarray(command.torque_body_nm, dtype=float),
                dt_s=0.5 * float(dt_s),
                acceleration_mode=self.acceleration_mode,
                numeric_backend=self.attitude_numeric_backend,
            )
        if self.drag_area_m2 is not None and "drag_area_m2" not in env_local:
            env_local["drag_area_m2"] = float(self.drag_area_m2)
        if self.lift_area_m2 is not None and "lift_area_m2" not in env_local:
            env_local["lift_area_m2"] = float(self.lift_area_m2)
        if self.srp_area_m2 is not None:
            env_local["srp_area_m2"] = float(self.srp_area_m2)
        area_profile = self.geometry_area_profile
        geom = self._rectangular_prism_geometry()
        base_env = dict(env_local)

        def stage_environment(
            position_eci_km: np.ndarray,
            velocity_eci_km_s: np.ndarray,
            t_s: float,
            quat_bn: np.ndarray,
        ) -> dict:
            """Build force inputs at the attitude and translational stage time.

            Directional projected areas must follow the attitude used by each
            orbit stage.  Reusing a full-step midpoint area for every orbit
            substep can apply the wrong force after a large attitude change.
            """

            stage_env = dict(base_env)
            if (
                self.lift_axis_body is not None
                and float(self.lift_coefficient) != 0.0
                and not bool(stage_env.get("physical_aerodynamics", False))
            ):
                c_bn = quaternion_to_dcm_bn(quat_bn)
                lift_axis_body = np.array(self.lift_axis_body, dtype=float).reshape(3)
                axis_norm = float(np.linalg.norm(lift_axis_body))
                if axis_norm > 0.0:
                    stage_env["lift_coefficient"] = float(self.lift_coefficient)
                    stage_env["lift_direction_eci"] = c_bn.T @ (lift_axis_body / axis_norm)

            if area_profile is None and not (
                self.use_rectangular_prism_for_aero_srp
                and geom is not None
                and self.disturbance_model is not None
            ):
                return stage_env

            c_bn = quaternion_to_dcm_bn(quat_bn)
            omega_raw = stage_env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            v_rel_eci_km_s = aero_core.atmosphere_relative_velocity_eci_km_s(
                position_eci_km,
                velocity_eci_km_s,
                t_s=float(t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(stage_env.get("drag_frame_model", "inertial_z")),
                jd_utc_start=stage_env.get("jd_utc_start"),
                eop_path=stage_env.get("drag_eop_path"),
                dut1_s=stage_env.get("dut1_s"),
                xp_arcsec=stage_env.get("xp_arcsec"),
                yp_arcsec=stage_env.get("yp_arcsec"),
                dat_s=stage_env.get("dat_s"),
                tt_minus_utc_s=stage_env.get("tt_minus_utc_s"),
                ddpsi_rad=float(stage_env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(stage_env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(stage_env.get("eop_extrapolation", "error") or "error"),
            )
            v_rel_body = c_bn @ v_rel_eci_km_s
            if area_profile is not None:
                stage_env["drag_area_m2"] = area_profile.projected_area_for_direction_m2(-v_rel_body)
                area_for_direction = area_profile.projected_area_for_direction_m2
            else:
                stage_env["drag_area_m2"] = geom.projected_area_m2(-v_rel_body)
                area_for_direction = geom.projected_area_m2

            srp_geometry = resolve_srp_geometry(position_eci_km, t_s, stage_env)
            sun_dir_eci = np.array(srp_geometry["sun_dir_sc_eci"], dtype=float)
            if float(np.linalg.norm(sun_dir_eci)) > 0.0:
                sun_dir_body = c_bn @ sun_dir_eci
                stage_env["srp_area_m2"] = area_for_direction(-sun_dir_body)
            return stage_env

        env_local = stage_environment(state.position_eci_km, state.velocity_eci_km_s, state.t_s, q_force_evaluation)

        x_orbit = np.empty(6, dtype=float)
        x_orbit[:3] = state.position_eci_km
        x_orbit[3:] = state.velocity_eci_km_s
        orbit_ctx = OrbitContext(
            mu_km3_s2=self.mu_km3_s2,
            mass_kg=state.mass_kg,
            area_m2=self.area_m2,
            cd=self.cd,
            cr=self.cr,
        )
        orbit_dt = self._effective_substep(self.orbit_substep_s, dt_s)
        command_acceleration_eci_km_s2 = np.asarray(command.thrust_eci_km_s2, dtype=float)
        physical_force_eci = command.mode_flags.get("physical_force_eci_n")
        physical_force_body = command.mode_flags.get("physical_force_body_n")
        if physical_force_eci is not None and physical_force_body is not None:
            c_bn_force = quaternion_to_dcm_bn(q_force_evaluation)
            total_force_eci_n = np.asarray(physical_force_eci, dtype=float).reshape(3) + c_bn_force.T @ np.asarray(
                physical_force_body,
                dtype=float,
            ).reshape(3)
            command_acceleration_eci_km_s2 = total_force_eci_n / max(float(state.mass_kg), 1.0e-12) / 1.0e3
        q_command = np.asarray(state.attitude_quat_bn, dtype=float).copy()
        w_command = np.asarray(state.angular_rate_body_rad_s, dtype=float).copy()
        t_local = float(state.t_s)
        x_orbit_next = x_orbit.copy()
        for h in self._substep_sequence(dt_s, orbit_dt):
            q_stage = q_command
            if propagate_attitude:
                q_stage, _ = propagate_attitude_exponential_map(
                    quat_bn=q_command,
                    omega_body_rad_s=w_command,
                    inertia_kg_m2=self.inertia_kg_m2,
                    torque_body_nm=np.asarray(command.torque_body_nm, dtype=float),
                    dt_s=0.5 * float(h),
                    acceleration_mode=self.acceleration_mode,
                    numeric_backend=self.attitude_numeric_backend,
                )
            stage_env = stage_environment(
                x_orbit_next[:3],
                x_orbit_next[3:],
                t_local + 0.5 * float(h),
                q_stage,
            )
            x_orbit_next = self.orbit_propagator.propagate(
                x_eci=x_orbit_next,
                dt_s=h,
                t_s=t_local,
                command_accel_eci_km_s2=command_acceleration_eci_km_s2,
                env=stage_env,
                ctx=orbit_ctx,
            )
            if propagate_attitude:
                q_command, w_command = propagate_attitude_exponential_map(
                    quat_bn=q_command,
                    omega_body_rad_s=w_command,
                    inertia_kg_m2=self.inertia_kg_m2,
                    torque_body_nm=np.asarray(command.torque_body_nm, dtype=float),
                    dt_s=float(h),
                    acceleration_mode=self.acceleration_mode,
                    numeric_backend=self.attitude_numeric_backend,
                )
            t_local += float(h)

        q_next = state.attitude_quat_bn.copy()
        w_next = state.angular_rate_body_rad_s.copy()
        midpoint_truth = self._midpoint_translational_truth(
            state=state,
            x_orbit_next=x_orbit_next,
            dt_s=dt_s,
        )
        env_local = stage_environment(
            midpoint_truth.position_eci_km,
            midpoint_truth.velocity_eci_km_s,
            midpoint_truth.t_s,
            q_force_evaluation,
        )
        if propagate_attitude:
            disturbance_cfg = getattr(self.disturbance_model, "config", None)
            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_drag", False)):
                if "density_kg_m3" not in env_local:
                    env_local["density_kg_m3"] = density_from_model(
                        str(env_local.get("atmosphere_model", "exponential")).lower(),
                        midpoint_truth.position_eci_km,
                        midpoint_truth.t_s,
                        env=env_local,
                    )
                omega_raw = env_local.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
                v_rel_eci_km_s = aero_core.atmosphere_relative_velocity_eci_km_s(
                    midpoint_truth.position_eci_km,
                    midpoint_truth.velocity_eci_km_s,
                    t_s=float(midpoint_truth.t_s),
                    earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                    frame_model=str(env_local.get("drag_frame_model", "inertial_z")),
                    jd_utc_start=env_local.get("jd_utc_start"),
                    eop_path=env_local.get("drag_eop_path"),
                    dut1_s=env_local.get("dut1_s"),
                    xp_arcsec=env_local.get("xp_arcsec"),
                    yp_arcsec=env_local.get("yp_arcsec"),
                    dat_s=env_local.get("dat_s"),
                    tt_minus_utc_s=env_local.get("tt_minus_utc_s"),
                    ddpsi_rad=float(env_local.get("ddpsi_rad", 0.0) or 0.0),
                    ddeps_rad=float(env_local.get("ddeps_rad", 0.0) or 0.0),
                    eop_extrapolation=str(env_local.get("eop_extrapolation", "error") or "error"),
                )
                v_rel_eci_m_s = v_rel_eci_km_s * 1e3
                env_local["drag_v_rel_eci_m_s"] = v_rel_eci_m_s
                env_local["drag_v_rel_norm_m_s"] = float(np.linalg.norm(v_rel_eci_m_s))

            if self.disturbance_model is not None and bool(getattr(disturbance_cfg, "use_srp", False)):
                srp_geometry = resolve_srp_geometry(midpoint_truth.position_eci_km, midpoint_truth.t_s, env_local)
                env_local["sun_dir_eci_unit"] = np.asarray(srp_geometry["sun_dir_sc_eci"], dtype=float)
                env_local["srp_distance_scale"] = float(srp_geometry["distance_scale"])
                if "srp_shadow_factor" not in env_local:
                    env_local["srp_shadow_factor"] = srp_shadow_factor(
                        r_sc_eci_km=midpoint_truth.position_eci_km,
                        t_s=midpoint_truth.t_s,
                        env=env_local,
                        srp_geometry=srp_geometry,
                    )
            att_dt = self._effective_substep(self.attitude_substep_s, dt_s)
            attitude_substeps = np.asarray(self._substep_sequence(dt_s, att_dt), dtype=float)
            compiled_propagator = getattr(self.disturbance_model, "try_propagate_compiled", None)
            if type(self.disturbance_model) is DisturbanceTorqueModel:
                compiled_result = self.disturbance_model.try_propagate_compiled(
                    quat_bn=q_next,
                    omega_body_rad_s=w_next,
                    command_torque_body_nm=command.torque_body_nm,
                    position_eci_km=midpoint_truth.position_eci_km,
                    t_s=midpoint_truth.t_s,
                    env=env_local,
                    substeps_s=attitude_substeps,
                    acceleration_mode=self.acceleration_mode,
                    acceleration_enabled=self._acceleration_enabled,
                )
            elif callable(compiled_propagator):
                compiled_result = compiled_propagator(
                    quat_bn=q_next,
                    omega_body_rad_s=w_next,
                    command_torque_body_nm=command.torque_body_nm,
                    position_eci_km=midpoint_truth.position_eci_km,
                    t_s=midpoint_truth.t_s,
                    env=env_local,
                    substeps_s=attitude_substeps,
                    acceleration_mode=self.acceleration_mode,
                )
            else:
                compiled_result = None
            if compiled_result is not None:
                q_next, w_next = compiled_result
            else:
                att_state = StateTruth(
                    position_eci_km=np.array(midpoint_truth.position_eci_km, dtype=float),
                    velocity_eci_km_s=np.array(midpoint_truth.velocity_eci_km_s, dtype=float),
                    attitude_quat_bn=np.array(q_next, dtype=float),
                    angular_rate_body_rad_s=np.array(w_next, dtype=float),
                    mass_kg=float(state.mass_kg),
                    t_s=float(midpoint_truth.t_s),
                )
                for h in attitude_substeps:
                    att_state.attitude_quat_bn = q_next
                    att_state.angular_rate_body_rad_s = w_next
                    disturbance_torque = (
                        np.zeros(3)
                        if self.disturbance_model is None
                        else self.disturbance_model.total_torque_body_nm(att_state, env_local)
                    )
                    total_torque = command.torque_body_nm + disturbance_torque
                    q_next, w_next = propagate_attitude_exponential_map(
                        quat_bn=q_next,
                        omega_body_rad_s=w_next,
                        inertia_kg_m2=self.inertia_kg_m2,
                        torque_body_nm=total_torque,
                        dt_s=float(h),
                        acceleration_mode=self.acceleration_mode,
                        numeric_backend=self.attitude_numeric_backend,
                    )

        # Optional direct attitude state override for surrogate controller testing.
        if propagate_attitude:
            att_override = dict(command.mode_flags.get("attitude_state_override", {}) or {})
            if att_override:
                q_cmd = np.array(att_override.get("q_next_bn", q_next), dtype=float).reshape(-1)
                w_cmd = np.array(att_override.get("w_next_body_rad_s", w_next), dtype=float).reshape(-1)
                q_next, w_next = guard_attitude_state_output(q_cmd, w_cmd)
        delta_mass_kg = float(command.mode_flags.get("delta_mass_kg", 0.0))
        min_mass_kg = float(command.mode_flags.get("min_mass_kg", 0.0))
        if not np.isfinite(min_mass_kg):
            min_mass_kg = 0.0
        min_mass_kg = max(min_mass_kg, 0.0)
        mass_next = max(min_mass_kg, state.mass_kg - delta_mass_kg)

        return StateTruth(
            position_eci_km=x_orbit_next[:3],
            velocity_eci_km_s=x_orbit_next[3:],
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=mass_next,
            t_s=state.t_s + dt_s,
        )

    def _native_mechanical_context(self, command: Command, env: dict):
        """Select the exact fused point-mass/GG/magnetic split-step envelope."""
        prop = self.orbit_propagator
        if (
            type(self) is not OrbitalAttitudeDynamics or type(prop) is not OrbitPropagator
            or self.attitude_numeric_backend != "rust" or prop.numeric_backend != "rust"
            or prop.model != "two_body" or prop.integrator != "rk4" or prop.plugins
            or not self.propagate_attitude or bool(env.get("attitude_disabled", False))
            or self.geometry_area_profile is not None or self.use_rectangular_prism_for_aero_srp
            or self.lift_axis_body is not None
        ):
            return None
        if command.mode_flags:
            # Metadata does not change point-mass mechanics. Physical wrench
            # inputs and overrides retain the existing coupled/Python owners.
            flags = command.mode_flags
            if dict(flags.get("attitude_state_override", {}) or {}):
                return None
            force_eci = np.asarray(flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float)
            force_body = np.asarray(flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float)
            if np.any(force_eci) or np.any(force_body) or float(flags.get("mass_flow_kg_s", 0.0) or 0.0) > 0.0:
                return None
        model = self.disturbance_model
        gravity_gradient = False
        magnetic = False
        if model is not None:
            if type(model) is not DisturbanceTorqueModel:
                return None
            cfg = model.config
            if (
                not model._compiled_plan_supported
                or cfg.numeric_backend != "rust"
                or cfg.use_drag or cfg.use_srp
                or float(model.mu_km3_s2) != float(self.mu_km3_s2)
                or not np.array_equal(model._compiled_inertia, self.inertia_kg_m2)
            ):
                return None
            gravity_gradient = bool(model._compiled_enabled[0])
            magnetic = bool(model._compiled_enabled[1])
        from sim.rust_orbit_backend import _extension

        constructor = getattr(_extension(), "MechanicalContext", None)
        if constructor is None:
            return None
        inertia = np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3)
        key = (float(self.mu_km3_s2), inertia.tobytes(), gravity_gradient)
        if magnetic:
            constructor = getattr(constructor, "with_magnetic", None)
            if constructor is None:
                return None
            dipole = np.asarray(model.config.magnetic_dipole_body_a_m2, dtype=float).reshape(3)
            field = np.asarray(env.get("magnetic_field_eci_t", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
            if not np.all(np.isfinite(dipole)) or not np.all(np.isfinite(field)):
                return None
            provided = "magnetic_field_eci_t" in env
            key += (dipole.tobytes(), field.tobytes(), provided)
        cached = getattr(prop, "_rust_mechanical_context_cache", None)
        if cached is None or cached[0] != key:
            native = (constructor(float(self.mu_km3_s2), inertia.ravel().tolist(), gravity_gradient,
                                  dipole.tolist(), field.tolist(), provided)
                      if magnetic else constructor(float(self.mu_km3_s2), inertia.ravel().tolist(), gravity_gradient))
            cached = (key, native)
            prop._rust_mechanical_context_cache = cached
        return cached[1]

    def try_native_mechanical_history(
        self, state: StateTruth, command: Command, env: dict, widths_s: np.ndarray,
    ) -> np.ndarray | None:
        """Pure numerical history for a caller-owned constant-command interval.

        This API executes no scenario lifecycle, controller or event boundaries.
        The ordinary step method uses the same native numerical owner.
        """
        if self.resource_model is not None:
            # Resource evolution belongs to step's outer interval owner.
            return None
        if command.mode_flags:
            # This numerical history does not carry mass or override telemetry.
            return None
        native = self._native_mechanical_context(command, env)
        if native is None:
            return None
        from sim.dynamics.attitude.rigid_body import _add_guardrail_counts

        vector = np.concatenate((state.position_eci_km, state.velocity_eci_km_s,
                                 state.attitude_quat_bn, state.angular_rate_body_rad_s))
        raw, counts = native.history(
            vector.tolist(), np.asarray(widths_s, dtype="<f8").tobytes(),
            self.orbit_substep_s, self.attitude_substep_s,
            np.asarray(command.thrust_eci_km_s2, dtype=float).tolist(),
            np.asarray(command.torque_body_nm, dtype=float).tolist(),
        )
        _add_guardrail_counts(np.asarray(counts, dtype=np.int64))
        self.orbit_propagator.last_numeric_path = "rust_native_mechanical_history"
        return np.frombuffer(raw, dtype="<f8").copy().reshape(-1, 13)

    def _try_native_coupled_step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth | None:
        """Exact native owner for point-mass physical wrench and GG/magnetic torques."""
        prop = self.orbit_propagator
        if (type(self) is not OrbitalAttitudeDynamics or type(prop) is not OrbitPropagator
                or self.attitude_numeric_backend != "rust" or prop.numeric_backend != "rust"
                or prop.model != "two_body" or prop.integrator != "rk4" or prop.state_frame != "eci"
                or self.geometry_area_profile is not None
                or self.use_rectangular_prism_for_aero_srp or self.lift_axis_body is not None
                or not self.propagate_attitude or bool(env.get("attitude_disabled", False))
                or dt_s <= 0.0 or not np.isfinite(dt_s)
                or not np.isfinite(self.mu_km3_s2) or self.mu_km3_s2 <= 0.0
                or not np.isfinite(state.t_s + dt_s)
                or dict(command.mode_flags.get("attitude_state_override", {}) or {})):
            return None
        model = self.disturbance_model
        enabled = np.zeros(4, dtype=np.int64)
        dipole = np.zeros(3, dtype=float)
        if model is not None:
            if type(model) is not DisturbanceTorqueModel:
                return None
            cfg = model.config
            if (not model._compiled_plan_supported or cfg.numeric_backend != "rust"
                    or (not prop.plugins and (cfg.use_drag or cfg.use_srp)) or float(model.mu_km3_s2) != float(self.mu_km3_s2)
                    or not np.array_equal(model.inertia_kg_m2, self.inertia_kg_m2)):
                return None
            enabled[:] = model._compiled_enabled
            dipole = np.asarray(cfg.magnetic_dipole_body_a_m2, dtype=float).reshape(3)
        inertia = np.asarray(self.inertia_kg_m2, dtype=float).reshape(3, 3)
        force_eci = np.asarray(command.mode_flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        force_body = np.asarray(command.mode_flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        torque = np.asarray(command.torque_body_nm, dtype=float).reshape(3)
        field = np.asarray(env.get("magnetic_field_eci_t", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        mass_flow = max(float(command.mode_flags.get("mass_flow_kg_s", 0.0)), 0.0)
        minimum_mass = max(float(command.mode_flags.get("min_mass_kg", 0.0)), 0.0)
        vector = np.concatenate((state.position_eci_km, state.velocity_eci_km_s,
                                 state.attitude_quat_bn, state.angular_rate_body_rad_s,
                                 np.asarray([state.mass_kg], dtype=float)))
        if (not all(np.all(np.isfinite(value)) for value in (vector, inertia, force_eci, force_body, torque, dipole, field))
                or not np.isfinite(mass_flow) or not np.isfinite(minimum_mass) or not np.isfinite(state.t_s)
                or state.mass_kg <= 0.0 or abs(float(np.linalg.norm(state.attitude_quat_bn)) - 1.0) > 1e-10
                or np.any(np.abs(state.angular_rate_body_rad_s) > 1e6) or np.any(np.abs(torque) > 1e12)):
            return None
        orbit_step = self._effective_substep(self.orbit_substep_s, dt_s)
        attitude_step = self._effective_substep(self.attitude_substep_s, dt_s)
        if not np.any(torque) and not np.any(state.angular_rate_body_rad_s) and not np.any(enabled):
            attitude_step = orbit_step
        maximum = min(orbit_step, attitude_step)
        from sim.rust_orbit_backend import _extension

        constructor = getattr(_extension(), "MechanicalContext", None)
        if constructor is None or not hasattr(constructor, "coupled_step"):
            return None
        key = (float(self.mu_km3_s2), inertia.tobytes(), False)
        cached = getattr(prop, "_rust_mechanical_context_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, constructor(float(self.mu_km3_s2), inertia.ravel().tolist(), False))
            prop._rust_mechanical_context_cache = cached
        if prop.plugins:
            function = getattr(cached[1], "coupled_force_step", None)
            if function is None or model is None:
                return None
            inputs = self._native_coupled_force_inputs(state, env, model)
            if inputs is None:
                return None
            force_context, disturbance_context, stage_callback = inputs
            values, counts = function(
                vector.tolist(), float(state.t_s), float(dt_s), float(maximum), force_eci.tolist(), force_body.tolist(),
                mass_flow, minimum_mass, torque.tolist(), force_context, disturbance_context, stage_callback,
            )
        else:
            values, counts = cached[1].coupled_step(
                vector.tolist(), float(state.t_s), float(dt_s), float(maximum), force_eci.tolist(), force_body.tolist(),
                mass_flow, minimum_mass, torque.tolist(), enabled.tolist(), dipole.tolist(), field.tolist(),
                "magnetic_field_eci_t" in env,
            )
        from sim.dynamics.attitude.rigid_body import _add_guardrail_counts

        _add_guardrail_counts(counts)
        result = np.asarray(values, dtype=float)
        prop.last_numeric_path = "rust_native_coupled_force_plan" if prop.plugins else "rust_native_coupled_mechanical"
        return StateTruth(result[:3].copy(), result[3:6].copy(), result[6:10].copy(), result[10:13].copy(),
                          float(result[13]), float(state.t_s + dt_s))

    def _native_coupled_force_inputs(self, state: StateTruth, env: dict, model: DisturbanceTorqueModel):
        """Retain native coefficients while visiting each authoritative stage environment."""
        if self.rectangular_prism_dims_m is not None:
            return None
        from sim.dynamics.orbit.environment import srp_pressure_n_m2
        from sim.dynamics.orbit.rust_force_plan import _CODES, make_plan, native_force_plan_codes
        from sim.rust_orbit_backend import native_harmonic_degree_limit

        prop = self.orbit_propagator
        if any(plugin not in _CODES for plugin in prop.plugins) or not model._refresh_mutable_facet_staging():
            return None
        config = model.config
        static_vectors = (model._compiled_magnetic_dipole, model._compiled_drag_cp_offset,
                          model._compiled_drag_facet_normals, model._compiled_drag_facet_areas,
                          model._compiled_drag_facet_cd, model._compiled_drag_facet_cp_offsets,
                          model._compiled_srp_cp_offset, model._compiled_srp_facet_normals,
                          model._compiled_srp_facet_areas, model._compiled_srp_facet_cp_offsets,
                          config.sun_dir_eci, env.get("sun_dir_eci_unit", env.get("sun_dir_eci", config.sun_dir_eci)))
        if (not all(np.all(np.isfinite(value)) for value in static_vectors)
                or not all(np.isfinite(value) and value >= 0.0 for value in (
                    config.drag_area_m2, config.drag_cd, config.srp_area_m2, config.srp_cr))):
            return None
        base_env = dict(env)
        if self.drag_area_m2 is not None:
            base_env["drag_area_m2"] = float(self.drag_area_m2)
        if self.lift_area_m2 is not None:
            base_env["lift_area_m2"] = float(self.lift_area_m2)
        if self.srp_area_m2 is not None:
            base_env["srp_area_m2"] = float(self.srp_area_m2)
        initial6 = np.concatenate((state.position_eci_km, state.velocity_eci_km_s))
        context = OrbitContext(self.mu_km3_s2, state.mass_kg, self.area_m2, self.cd, self.cr)
        if native_force_plan_codes(prop, base_env) is None:
            return None
        if any(_CODES[plugin] == 1 for plugin in prop.plugins):
            from sim.dynamics.orbit.propagator import spherical_harmonics_plugin

            if base_env.get("_compiled_spherical_harmonics_terms") is None:
                spherical_harmonics_plugin(state.t_s, initial6, base_env, context)
            harmonic = base_env.get("_compiled_spherical_harmonics_terms")
            if (harmonic is None or not harmonic.all_normalized
                    or harmonic.n_max > native_harmonic_degree_limit()):
                return None
        codes, _, _, dimensions, _, force_stage, force_context = make_plan(prop, initial6, state.t_s, base_env, context)
        if (force_context is None or (1 in codes and dimensions[0] > native_harmonic_degree_limit())):
            return None
        disturbance_context = model._native_disturbance_context()
        if disturbance_context is None:
            return None
        field = np.asarray(env.get("magnetic_field_eci_t", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist()
        provided = float("magnetic_field_eci_t" in env)
        shared_drag_frame = str(base_env.get("drag_frame_model", "inertial_z")) == str(base_env.get("drag_frame_model", "simple"))

        def stage_callback(t_s, raw_state):
            x = np.asarray(raw_state, dtype=float).reshape(6)
            result = list(force_stage(t_s, raw_state))
            stage_env = dict(base_env)
            epoch = resolve_time_dependent_env(base_env, t_s)
            for key in ("sun_pos_eci_km", "moon_pos_eci_km"):
                if key not in stage_env and key in epoch:
                    stage_env[key] = epoch[key]
            density = 0.0
            wind = np.zeros(3)
            wind_norm = 0.0
            if config.use_drag:
                if "density_kg_m3" in stage_env:
                    density = float(stage_env["density_kg_m3"])
                elif 2 in codes:
                    density = float(result[9])
                else:
                    density = density_from_model(str(stage_env.get("atmosphere_model", "exponential")).lower(), x[:3], t_s, env=stage_env)
                if "drag_v_rel_eci_m_s" in stage_env:
                    wind = np.asarray(stage_env["drag_v_rel_eci_m_s"], dtype=float).reshape(3)
                elif 2 in codes and shared_drag_frame:
                    wind = np.asarray(result[10:13], dtype=float) * 1e3
                else:
                    omega_raw = stage_env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
                    wind = aero_core.atmosphere_relative_velocity_eci_km_s(
                        x[:3], x[3:], t_s=t_s,
                        earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                        frame_model=str(stage_env.get("drag_frame_model", "inertial_z")),
                        jd_utc_start=stage_env.get("jd_utc_start"), eop_path=stage_env.get("drag_eop_path"),
                        dut1_s=stage_env.get("dut1_s"), xp_arcsec=stage_env.get("xp_arcsec"), yp_arcsec=stage_env.get("yp_arcsec"),
                        dat_s=stage_env.get("dat_s"), tt_minus_utc_s=stage_env.get("tt_minus_utc_s"),
                        ddpsi_rad=float(stage_env.get("ddpsi_rad", 0.0) or 0.0), ddeps_rad=float(stage_env.get("ddeps_rad", 0.0) or 0.0),
                        eop_extrapolation=str(stage_env.get("eop_extrapolation", "error") or "error"),
                    ) * 1e3
                wind_norm = float(stage_env["drag_v_rel_norm_m_s"]) if "drag_v_rel_eci_m_s" in stage_env and "drag_v_rel_norm_m_s" in stage_env else float(np.sqrt(np.dot(wind, wind)))
            sun = np.zeros(3)
            pressure = 0.0
            if config.use_srp:
                sun = np.asarray(stage_env.get("sun_dir_eci_unit", stage_env.get("sun_dir_eci", config.sun_dir_eci)), dtype=float).reshape(3)
                norm = float(np.sqrt(np.dot(sun, sun)))
                if norm > 0.0:
                    if "sun_dir_eci_unit" not in stage_env:
                        sun = sun / norm
                    shadow = float(stage_env["srp_shadow_factor"]) if "srp_shadow_factor" in stage_env else float(srp_shadow_factor(x[:3], t_s, stage_env))
                    if shadow > 0.0:
                        pressure = srp_pressure_n_m2(stage_env) * float(stage_env.get("srp_distance_scale", 1.0)) * config.srp_cr * shadow
            return result + field + [provided, float(density)] + wind.tolist() + [wind_norm] + sun.tolist() + [float(pressure)]

        return force_context, disturbance_context, stage_callback

    def _step_coupled_v2(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        """Run the production v2 stage-consistent coupled physics path."""

        native = self._try_native_coupled_step(state, command, env, dt_s)
        if native is not None:
            return native

        force_eci_n = np.asarray(command.mode_flags.get("physical_force_eci_n", (0.0, 0.0, 0.0)), dtype=float)
        force_body_n = np.asarray(command.mode_flags.get("physical_force_body_n", (0.0, 0.0, 0.0)), dtype=float)
        commanded_torque = np.asarray(command.torque_body_nm, dtype=float)
        mass_flow_kg_s = max(float(command.mode_flags.get("mass_flow_kg_s", 0.0)), 0.0)
        minimum_mass_kg = max(float(command.mode_flags.get("min_mass_kg", 0.0)), 0.0)
        environment = dict(env)

        def derivative(t_s: float, stage: CoupledSatelliteState, _control: object) -> CoupledDerivative:
            propellant_available = stage.mass_kg > minimum_mass_kg + 1.0e-12
            active_force_eci = force_eci_n if propellant_available or mass_flow_kg_s == 0.0 else np.zeros(3)
            active_force_body = force_body_n if propellant_available or mass_flow_kg_s == 0.0 else np.zeros(3)
            active_mass_flow = mass_flow_kg_s if propellant_available else 0.0
            c_bn = quaternion_to_dcm_bn(stage.attitude_quat_bn)
            total_force_eci_n = active_force_eci + c_bn.T @ active_force_body
            stage_truth = StateTruth(
                position_eci_km=stage.position_eci_km,
                velocity_eci_km_s=stage.velocity_eci_km_s,
                attitude_quat_bn=stage.attitude_quat_bn,
                angular_rate_body_rad_s=stage.angular_rate_body_rad_s,
                mass_kg=stage.mass_kg,
                t_s=t_s,
            )
            stage_env = self._coupled_stage_environment(
                stage_truth,
                environment,
                attitude_dcm_bn=c_bn,
            )
            orbit_ctx = OrbitContext(
                mu_km3_s2=self.mu_km3_s2,
                mass_kg=stage.mass_kg,
                area_m2=self.area_m2,
                cd=self.cd,
                cr=self.cr,
            )
            state6 = np.hstack((stage.position_eci_km, stage.velocity_eci_km_s))
            acceleration = self.orbit_propagator.acceleration_at(
                t_s=t_s,
                x_eci=state6,
                command_accel_eci_km_s2=total_force_eci_n / stage.mass_kg / 1.0e3,
                env=stage_env,
                ctx=orbit_ctx,
            )
            disturbance = (
                np.zeros(3)
                if self.disturbance_model is None
                else self.disturbance_model.total_torque_body_nm(stage_truth, stage_env)
            )
            total_torque = commanded_torque + disturbance
            omega = stage.angular_rate_body_rad_s
            if self.attitude_numeric_backend == "rust":
                from sim.dynamics.attitude.rigid_body import rigid_body_derivatives

                _, angular_acceleration = rigid_body_derivatives(
                    stage.attitude_quat_bn, omega, self.inertia_kg_m2, total_torque,
                    numeric_backend="rust",
                )
            else:
                angular_acceleration = np.linalg.solve(
                    self.inertia_kg_m2,
                    total_torque - np.cross(omega, self.inertia_kg_m2 @ omega),
                )
            return CoupledDerivative(
                position_rate_km_s=stage.velocity_eci_km_s,
                velocity_rate_km_s2=acceleration,
                angular_acceleration_body_rad_s2=angular_acceleration,
                mass_rate_kg_s=-active_mass_flow,
                actuator_state_rate=np.empty(0, dtype=float),
            )

        orbit_step = self._effective_substep(self.orbit_substep_s, dt_s)
        attitude_step = self._effective_substep(self.attitude_substep_s, dt_s)
        if (
            not np.any(commanded_torque)
            and not np.any(state.angular_rate_body_rad_s)
            and (
                self.disturbance_model is None
                or not bool(getattr(self.disturbance_model, "_compiled_has_disturbances", True))
            )
        ):
            # A static attitude makes the body-to-inertial force transform
            # constant. Preserve the exact coupled equations while avoiding
            # needless attitude microsteps during long coasts/steady burns.
            attitude_step = orbit_step
        integrator = CoupledSatelliteIntegrator(
            CoupledIntegratorConfig(orbit_step, attitude_step),
            derivative,
            numeric_backend=self.orbit_propagator.numeric_backend,
        )
        initial = CoupledSatelliteState(
            position_eci_km=state.position_eci_km,
            velocity_eci_km_s=state.velocity_eci_km_s,
            attitude_quat_bn=state.attitude_quat_bn,
            angular_rate_body_rad_s=state.angular_rate_body_rad_s,
            mass_kg=state.mass_kg,
            actuator_state=np.empty(0, dtype=float),
            t_s=state.t_s,
        )
        final = integrator.propagate(initial, end_time_s=state.t_s + dt_s).final_state
        q_next = final.attitude_quat_bn
        w_next = final.angular_rate_body_rad_s
        attitude_override = dict(command.mode_flags.get("attitude_state_override", {}) or {})
        if attitude_override:
            q_next, w_next = guard_attitude_state_output(
                np.asarray(attitude_override.get("q_next_bn", q_next), dtype=float),
                np.asarray(attitude_override.get("w_next_body_rad_s", w_next), dtype=float),
            )
        return StateTruth(
            position_eci_km=final.position_eci_km,
            velocity_eci_km_s=final.velocity_eci_km_s,
            attitude_quat_bn=q_next,
            angular_rate_body_rad_s=w_next,
            mass_kg=max(float(final.mass_kg), minimum_mass_kg),
            t_s=final.t_s,
        )

    def _coupled_stage_environment(
        self,
        state: StateTruth,
        env: dict,
        *,
        attitude_dcm_bn: np.ndarray | None = None,
    ) -> dict:
        stage_env = dict(env)
        resolved_epoch = resolve_time_dependent_env(env, state.t_s)
        for key in ("sun_pos_eci_km", "moon_pos_eci_km"):
            if key not in stage_env and key in resolved_epoch:
                stage_env[key] = resolved_epoch[key]
        if self.drag_area_m2 is not None:
            stage_env["drag_area_m2"] = float(self.drag_area_m2)
        if self.lift_area_m2 is not None:
            stage_env["lift_area_m2"] = float(self.lift_area_m2)
        if self.srp_area_m2 is not None:
            stage_env["srp_area_m2"] = float(self.srp_area_m2)
        c_bn = (
            quaternion_to_dcm_bn(state.attitude_quat_bn)
            if attitude_dcm_bn is None
            else attitude_dcm_bn
        )
        if self.lift_axis_body is not None and float(self.lift_coefficient) != 0.0:
            axis = np.asarray(self.lift_axis_body, dtype=float).reshape(3)
            norm = float(np.linalg.norm(axis))
            if norm > 0.0:
                stage_env["lift_coefficient"] = float(self.lift_coefficient)
                stage_env["lift_direction_eci"] = c_bn.T @ (axis / norm)
        area_profile = self.geometry_area_profile
        geometry = self._rectangular_prism_geometry()
        if area_profile is not None or geometry is not None:
            omega_raw = stage_env.get("drag_earth_rotation_rad_s", EARTH_ROT_RATE_RAD_S)
            relative_velocity = aero_core.atmosphere_relative_velocity_eci_km_s(
                state.position_eci_km,
                state.velocity_eci_km_s,
                t_s=float(state.t_s),
                earth_rotation_rad_s=float(EARTH_ROT_RATE_RAD_S if omega_raw is None else omega_raw),
                frame_model=str(stage_env.get("drag_frame_model", "inertial_z")),
                jd_utc_start=stage_env.get("jd_utc_start"),
                eop_path=stage_env.get("drag_eop_path"),
                dut1_s=stage_env.get("dut1_s"),
                xp_arcsec=stage_env.get("xp_arcsec"),
                yp_arcsec=stage_env.get("yp_arcsec"),
                dat_s=stage_env.get("dat_s"),
                tt_minus_utc_s=stage_env.get("tt_minus_utc_s"),
                ddpsi_rad=float(stage_env.get("ddpsi_rad", 0.0) or 0.0),
                ddeps_rad=float(stage_env.get("ddeps_rad", 0.0) or 0.0),
                eop_extrapolation=str(stage_env.get("eop_extrapolation", "error") or "error"),
            )
            body_velocity = c_bn @ relative_velocity
            area_owner = area_profile if area_profile is not None else geometry
            stage_env["drag_area_m2"] = area_owner.projected_area_for_direction_m2(-body_velocity) if area_profile is not None else area_owner.projected_area_m2(-body_velocity)
            srp_geometry = resolve_srp_geometry(state.position_eci_km, state.t_s, stage_env)
            sun_direction = np.asarray(srp_geometry["sun_dir_sc_eci"], dtype=float)
            if float(np.linalg.norm(sun_direction)) > 0.0:
                body_sun = c_bn @ sun_direction
                stage_env["srp_area_m2"] = area_owner.projected_area_for_direction_m2(-body_sun) if area_profile is not None else area_owner.projected_area_m2(-body_sun)
        return stage_env

    def _rectangular_prism_geometry(self) -> RectangularPrismGeometry | None:
        dims = self.rectangular_prism_dims_m
        if dims is None:
            return None
        return RectangularPrismGeometry(lx_m=float(dims[0]), ly_m=float(dims[1]), lz_m=float(dims[2]))

    @staticmethod
    def _effective_substep(substep_s: float | None, dt_s: float) -> float:
        if substep_s is None:
            return dt_s
        return max(min(float(substep_s), dt_s), 1e-9)

    @staticmethod
    def _substep_sequence(total_dt_s: float, h_s: float) -> list[float]:
        if h_s >= total_dt_s:
            return [float(total_dt_s)]
        n = int(np.floor(total_dt_s / h_s))
        steps = [float(h_s)] * n
        rem = float(total_dt_s - n * h_s)
        if rem > 1e-12:
            steps.append(rem)
        return steps

    @staticmethod
    def _midpoint_translational_truth(state: StateTruth, x_orbit_next: np.ndarray, dt_s: float) -> StateTruth:
        x_orbit_now = np.empty(6, dtype=float)
        x_orbit_now[:3] = state.position_eci_km
        x_orbit_now[3:] = state.velocity_eci_km_s
        x_mid = 0.5 * (x_orbit_now + np.array(x_orbit_next, dtype=float).reshape(6))
        return StateTruth(
            position_eci_km=np.array(x_mid[:3], dtype=float),
            velocity_eci_km_s=np.array(x_mid[3:], dtype=float),
            attitude_quat_bn=np.array(state.attitude_quat_bn, dtype=float),
            angular_rate_body_rad_s=np.array(state.angular_rate_body_rad_s, dtype=float),
            mass_kg=float(state.mass_kg),
            t_s=float(state.t_s + 0.5 * dt_s),
        )
