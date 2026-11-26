"""Passive rigid-body rotation driven by a prepared primary-centered ONP orbit."""

from __future__ import annotations

import math

import numpy as np

from sim.digital_twin.mass_properties import resolve_inertia_kg_m2
from sim.dynamics.attitude.disturbances import centered_dipole_field_t, gravity_gradient_torque_body_nm
from sim.dynamics.attitude.rigid_body import guard_attitude_state_output, rigid_body_derivatives
from sim.dynamics.orbit.integrators import combine_adaptive_step_info, rk4_step_state
from sim.utils.quaternion import quaternion_to_dcm_bn


class PlanetaryAttitude:
    def __init__(self, plan, specs, settings):
        self.plan = plan
        self.environment = plan.environment
        self.inertia = resolve_inertia_kg_m2(specs)
        # Euler's equations are homogeneous in inertia and applied torque.
        # Conditioning both together also accommodates moon-scale moments.
        self.inertia_scale = np.max(np.diag(self.inertia))
        self.scaled_inertia = self.inertia / self.inertia_scale
        self.backend = settings.get("numeric_backend", "rust")
        self.substep_s = settings.get("attitude_substep_s")
        selected = settings.get("disturbance_torques", {})
        self.gravity_gradient = selected.get("gravity_gradient", False)
        self.pressure_offsets = {
            name: np.asarray(specs[f"{name}_cp_offset_body_m"], dtype=float)
            for name in ("drag", "srp")
            if selected.get(name)
        }
        self.spacecraft_dipole = (
            np.asarray(specs["magnetic_dipole_body_a_m2"], dtype=float) if selected.get("magnetic") else None
        )

    def torques(self, t_s, orbit, quat, forces=None):
        """Return instantaneous body-axis torques about the mass center in N m."""
        c_bn = quaternion_to_dcm_bn(quat)
        components = {}
        if self.gravity_gradient:
            components["gravity_gradient"] = gravity_gradient_torque_body_nm(
                self.environment.gravity.mu_km3_s2, self.inertia, orbit[:3], c_bn
            )
        if self.pressure_offsets:
            if forces is None:
                forces = self.plan.stage(t_s, orbit)[1]["forces_km_s2"]
            for name, offset in self.pressure_offsets.items():
                force_n = np.asarray(forces[name]) * (1000.0 * self.plan.mass_kg)
                components[name] = np.cross(offset, c_bn @ force_n)
        if self.spacecraft_dipole is not None:
            rotation, _ = self.environment.gravity_frame.evaluate(t_s)
            dipole = self.environment.selections["magnetic_field"]["dipole_t_m3"]
            field_fixed = centered_dipole_field_t(rotation @ orbit[:3], np.asarray(dipole))
            components["magnetic"] = np.cross(self.spacecraft_dipole, c_bn @ (rotation.T @ field_fixed))
        return components

    def propagate(self, propagator, state, dt_s):
        orbit = np.r_[state.position_eci_km, state.velocity_eci_km_s]
        attitude = np.r_[state.attitude_quat_bn, state.angular_rate_body_rad_s]
        if dt_s == 0:
            return orbit, attitude
        count = max(1, math.ceil(dt_s / (self.substep_s or dt_s) - 1e-12))
        step = dt_s / count
        adaptive_steps = []
        for index in range(count):
            start = state.t_s + index * step
            half = step * 0.5
            mid_orbit = propagator.propagate(orbit, half, start)
            if propagator.last_adaptive_info is not None:
                adaptive_steps.append(propagator.last_adaptive_info)
            end_orbit = propagator.propagate(mid_orbit, half, start + half)
            if propagator.last_adaptive_info is not None:
                adaptive_steps.append(propagator.last_adaptive_info)
            stages = {0.0: orbit, half: mid_orbit, step: end_orbit}

            def derivative(offset, y):
                components = self.torques(start + offset, stages[offset], y[:4])
                torque = sum(components.values(), np.zeros(3))
                dq, dw = rigid_body_derivatives(
                    y[:4],
                    y[4:],
                    self.scaled_inertia,
                    torque / self.inertia_scale,
                    numeric_backend=self.backend,
                )
                # The shared kernel returns qdot for a unit quaternion. Restore
                # its linear dependence on the unnormalized RK stage quaternion.
                return np.r_[dq * np.linalg.norm(y[:4]), dw]

            attitude = rk4_step_state(derivative, 0.0, attitude, step)
            attitude[:4], attitude[4:] = guard_attitude_state_output(attitude[:4], attitude[4:])
            orbit = end_orbit
        if adaptive_steps:
            propagator.last_adaptive_info = combine_adaptive_step_info(propagator.integrator, adaptive_steps)
        return orbit, attitude
