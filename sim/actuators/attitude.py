from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.quaternion import quaternion_to_dcm_bn


@dataclass(frozen=True)
class ReactionWheelLimits:
    max_torque_nm: np.ndarray
    max_momentum_nms: np.ndarray
    wheel_axes_body: np.ndarray | None = None  # shape (3,N) or (N,3)
    wheel_inertia_kg_m2: np.ndarray | float | None = None
    max_speed_rad_s: np.ndarray | float | None = None
    torque_time_constant_s: float = 0.0
    viscous_friction_nms: np.ndarray | float = 0.0
    coulomb_friction_nm: np.ndarray | float = 0.0
    numeric_backend: str = "rust"


@dataclass(frozen=True)
class MagnetorquerLimits:
    max_dipole_a_m2: np.ndarray


@dataclass(frozen=True)
class ThrusterPulseLimits:
    max_torque_nm: np.ndarray
    pulse_quantum_s: float = 0.02


@dataclass(frozen=True)
class ControlMomentGyroLimits:
    max_torque_nm: np.ndarray | float
    momentum_nms: np.ndarray | float
    gimbal_rate_limit_rad_s: np.ndarray | float = np.inf
    torque_time_constant_s: float = 0.0


@dataclass(frozen=True)
class WheelDesaturationLimits:
    momentum_fraction_threshold: float = 0.8
    unload_gain_s_inv: float = 0.02
    max_unload_torque_nm: float = 0.01


@dataclass
class AttitudeActuator(Actuator):
    reaction_wheels: ReactionWheelLimits | None = None
    magnetorquers: MagnetorquerLimits | None = None
    thruster_pulse: ThrusterPulseLimits | None = None
    control_moment_gyros: ControlMomentGyroLimits | None = None
    wheel_desaturation: WheelDesaturationLimits | None = None
    # Body-equivalent wheel momentum vector (N*m*s) for telemetry/compatibility.
    wheel_momentum_nms: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Per-wheel internal states (allocated lazily from wheel config).
    wheel_momentum_wheels_nms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wheel_speed_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    wheel_motor_torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cmg_torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _wheel_allocator_signature: tuple[tuple[int, int], tuple[float, ...]] | None = field(
        default=None, init=False, repr=False
    )
    _wheel_allocator_pinv: np.ndarray | None = field(default=None, init=False, repr=False)
    _wheel_staging_cache: tuple | None = field(default=None, init=False, repr=False)
    _wheel_native_prepared: bytes | None = field(default=None, init=False, repr=False)

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        dt_s = float(dt_s)
        if not np.isfinite(dt_s):
            raise ValueError("dt_s must be finite")
        # Match the attitude propagator's guardrail: a stale timestamp cannot
        # integrate wheel momentum backward in time, so it becomes a zero step.
        dt_s = max(dt_s, 0.0)
        requested_torque = np.array(command.torque_body_nm, dtype=float)
        if requested_torque.shape != (3,) or not np.all(np.isfinite(requested_torque)):
            raise ValueError("command.torque_body_nm must be a finite three-vector")
        allocated_torque = np.zeros(3, dtype=float)
        auxiliary_torque = np.zeros(3, dtype=float)
        has_primary_device = False
        mode_flags = dict(command.mode_flags)
        pre_step_wheel_momentum_nms = np.array(self.wheel_momentum_nms, dtype=float).reshape(3)

        if self.reaction_wheels is not None:
            wheel_torque, rw_diag = self._apply_reaction_wheels(
                torque_body_cmd_nm=requested_torque - allocated_torque,
                dt_s=dt_s,
                mode_flags=mode_flags,
            )
            allocated_torque += wheel_torque
            has_primary_device = True
            mode_flags.update(rw_diag)

        if self.wheel_desaturation is not None and self.reaction_wheels is not None:
            h_now = np.array(self.wheel_momentum_nms, dtype=float).reshape(3)
            if np.linalg.norm(pre_step_wheel_momentum_nms) > np.linalg.norm(h_now):
                h_now = pre_step_wheel_momentum_nms
            unload_torque, desat_diag = self._desaturation_torque(h_now)
            auxiliary_torque += unload_torque
            mode_flags.update(desat_diag)

        if self.control_moment_gyros is not None:
            cmg_torque, cmg_diag = self._apply_control_moment_gyros(
                requested_torque - allocated_torque,
                dt_s=dt_s,
            )
            allocated_torque += cmg_torque
            has_primary_device = True
            mode_flags.update(cmg_diag)

        if self.thruster_pulse is not None:
            tp = self.thruster_pulse
            pulse_torque = np.clip(
                requested_torque - allocated_torque,
                -tp.max_torque_nm,
                tp.max_torque_nm,
            )
            if tp.pulse_quantum_s > 0.0:
                pulses = np.round(dt_s / tp.pulse_quantum_s)
                scale = 0.0 if pulses <= 0 else pulses * tp.pulse_quantum_s / dt_s
                pulse_torque *= min(float(scale), 1.0)
            allocated_torque += pulse_torque
            has_primary_device = True

        if self.magnetorquers is not None:
            mt_torque, mt_diag = self._apply_magnetorquers(
                requested_torque - allocated_torque,
                mode_flags,
            )
            allocated_torque += mt_torque
            has_primary_device = True
            mode_flags.update(mt_diag)

        torque = (allocated_torque if has_primary_device else requested_torque) + auxiliary_torque

        return Command(
            thrust_eci_km_s2=np.array(command.thrust_eci_km_s2, dtype=float),
            torque_body_nm=torque,
            mode_flags=mode_flags,
        )

    @staticmethod
    def _as_vector(value: np.ndarray | float | None, n: int, default: float = 0.0) -> np.ndarray:
        if value is None:
            return np.full(n, float(default), dtype=float)
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=float)
        if arr.size != n:
            raise ValueError(f"Expected scalar or length-{n} vector, got length {arr.size}.")
        return arr

    @staticmethod
    def _resolve_wheel_axes(wheel_axes_body: np.ndarray | None, n: int) -> np.ndarray:
        if wheel_axes_body is None:
            if n == 3:
                return np.eye(3, dtype=float)
            raise ValueError("wheel_axes_body is required when number of wheels is not 3.")
        axes = np.array(wheel_axes_body, dtype=float)
        if axes.ndim != 2:
            raise ValueError("wheel_axes_body must be a 2D array.")
        if axes.shape[0] == 3:
            g = axes.copy()
        elif axes.shape[1] == 3:
            g = axes.T.copy()
        else:
            raise ValueError("wheel_axes_body must have shape (3,N) or (N,3).")
        if g.shape[1] != n:
            raise ValueError(f"wheel_axes_body must contain {n} wheel axes.")
        for k in range(n):
            norm = float(np.linalg.norm(g[:, k]))
            if norm <= 0.0:
                raise ValueError("wheel_axes_body contains a zero vector.")
            g[:, k] /= norm
        return g

    def _apply_reaction_wheels(
        self,
        torque_body_cmd_nm: np.ndarray,
        dt_s: float,
        mode_flags: dict,
    ) -> tuple[np.ndarray, dict[str, object]]:
        rw = self.reaction_wheels
        if rw is None:
            return torque_body_cmd_nm, {}
        backend = normalize_numeric_backend(rw.numeric_backend, error_message="reaction-wheel numeric_backend must be python or rust")

        n_wheels = int(np.array(rw.max_torque_nm, dtype=float).reshape(-1).size)
        if n_wheels <= 0:
            return np.zeros(3, dtype=float), {}

        signature = tuple(
            None if value is None else (np.asarray(value, dtype=float).shape,
                                       np.asarray(value, dtype=float).tobytes())
            for value in (rw.wheel_axes_body, rw.max_torque_nm, rw.max_momentum_nms,
                          rw.max_speed_rad_s, rw.viscous_friction_nms,
                          rw.coulomb_friction_nm, rw.wheel_inertia_kg_m2)
        )
        cached = self._wheel_staging_cache
        if cached is None or cached[0] != signature:
            g = self._resolve_wheel_axes(rw.wheel_axes_body, n=n_wheels)
            max_torque_nm = np.abs(self._as_vector(rw.max_torque_nm, n=n_wheels, default=0.0))
            max_momentum_nms = np.abs(self._as_vector(rw.max_momentum_nms, n=n_wheels, default=np.inf))
            max_speed_rad_s = np.abs(self._as_vector(rw.max_speed_rad_s, n=n_wheels, default=np.inf))
            viscous_nms = np.abs(self._as_vector(rw.viscous_friction_nms, n=n_wheels, default=0.0))
            coulomb_nm = np.abs(self._as_vector(rw.coulomb_friction_nm, n=n_wheels, default=0.0))

            # Derive wheel inertia from explicit input or from h_max / w_max when available.
            j_kg_m2 = self._as_vector(rw.wheel_inertia_kg_m2, n=n_wheels, default=np.nan)
            inferred_mask = ~np.isfinite(j_kg_m2)
            if np.any(inferred_mask):
                infer_ok = np.isfinite(max_momentum_nms) & np.isfinite(max_speed_rad_s) & (max_speed_rad_s > 0.0)
                j_kg_m2[inferred_mask & infer_ok] = (
                    max_momentum_nms[inferred_mask & infer_ok] / max_speed_rad_s[inferred_mask & infer_ok]
                )
                j_kg_m2[~np.isfinite(j_kg_m2)] = 5e-4
            j_kg_m2 = np.clip(j_kg_m2, 1e-9, np.inf)
            cached = (signature, g, max_torque_nm, max_momentum_nms, max_speed_rad_s,
                      viscous_nms, coulomb_nm, j_kg_m2)
            self._wheel_staging_cache = cached
            prepared_values = (g, max_torque_nm, max_momentum_nms, max_speed_rad_s,
                               viscous_nms, coulomb_nm, j_kg_m2)
            self._wheel_native_prepared = (
                b"".join(np.asarray(value, dtype="<f8").tobytes() for value in prepared_values)
                if all(not np.any(np.isnan(value)) for value in prepared_values)
                and np.all(np.isfinite(j_kg_m2)) else None
            )
        _, g, max_torque_nm, max_momentum_nms, max_speed_rad_s, viscous_nms, coulomb_nm, j_kg_m2 = cached
        torque_tau_s = float(max(rw.torque_time_constant_s, 0.0))

        if self.wheel_speed_rad_s.size != n_wheels:
            self.wheel_speed_rad_s = np.zeros(n_wheels, dtype=float)
        if self.wheel_motor_torque_nm.size != n_wheels:
            self.wheel_motor_torque_nm = np.zeros(n_wheels, dtype=float)
        if self.wheel_momentum_wheels_nms.size != n_wheels:
            self.wheel_momentum_wheels_nms = np.zeros(n_wheels, dtype=float)

        wheel_torque_mode_flag = mode_flags.get("wheel_torque_cmd_nm", None)
        signature = (g.shape, tuple(float(value) for value in g.reshape(-1)))
        if signature != self._wheel_allocator_signature or self._wheel_allocator_pinv is None:
            self._wheel_allocator_signature = signature
            self._wheel_allocator_pinv = np.linalg.pinv(g)
        if wheel_torque_mode_flag is not None:
            tau_cmd = np.array(wheel_torque_mode_flag, dtype=float).reshape(-1)
            if tau_cmd.size != n_wheels:
                tau_cmd = -self._wheel_allocator_pinv @ np.array(torque_body_cmd_nm, dtype=float).reshape(3)
        else:
            tau_cmd = -self._wheel_allocator_pinv @ np.array(torque_body_cmd_nm, dtype=float).reshape(3)
        tau_cmd = np.clip(tau_cmd, -max_torque_nm, max_torque_nm)

        native_result = None
        if backend == "rust" and self._wheel_native_prepared is not None and np.isfinite(dt_s) and np.isfinite(torque_tau_s):
            from sim.rust_control_backend import reaction_wheel_step

            native_result = reaction_wheel_step(
                self._wheel_native_prepared, self.wheel_motor_torque_nm,
                self.wheel_speed_rad_s, tau_cmd, dt_s, torque_tau_s,
            )
        if native_result is not None:
            tau_motor_eff = native_result[:n_wheels]
            omega_next = native_result[n_wheels:2*n_wheels]
            h_next = native_result[2*n_wheels:3*n_wheels]
            realized_tau_net = native_result[3*n_wheels:4*n_wheels]
            self.wheel_momentum_nms = native_result[4*n_wheels:4*n_wheels+3]
            torque_body_nm = native_result[4*n_wheels+3:]
            self.wheel_motor_torque_nm = tau_motor_eff
            self.wheel_speed_rad_s = omega_next
            self.wheel_momentum_wheels_nms = h_next
        else:
            # First-order wheel motor torque lag.
            if dt_s <= 0.0 or torque_tau_s <= 0.0:
                tau_motor = tau_cmd
            else:
                alpha = float(np.clip(dt_s / torque_tau_s, 0.0, 1.0))
                tau_motor = self.wheel_motor_torque_nm + alpha * (tau_cmd - self.wheel_motor_torque_nm)
            tau_motor = np.clip(tau_motor, -max_torque_nm, max_torque_nm)
            tau_motor_eff = tau_motor.copy()

            # Friction torque opposes wheel spin and reduces achievable wheel acceleration.
            omega = np.array(self.wheel_speed_rad_s, dtype=float)
            tau_fric = viscous_nms * omega + coulomb_nm * np.sign(omega)
            tau_net = tau_motor_eff - tau_fric

            # Prevent driving further into momentum saturation.
            h_now = j_kg_m2 * omega
            sat_hi = (h_now >= (max_momentum_nms - 1e-12)) & (tau_net > 0.0)
            sat_lo = (h_now <= (-max_momentum_nms + 1e-12)) & (tau_net < 0.0)
            sat_h = sat_hi | sat_lo
            tau_net[sat_h] = 0.0
            tau_motor_eff[sat_h] = tau_fric[sat_h]

            omega_prop = omega + dt_s * (tau_net / j_kg_m2)
            sat_speed_hi = (omega_prop >= max_speed_rad_s) & (tau_net > 0.0)
            sat_speed_lo = (omega_prop <= -max_speed_rad_s) & (tau_net < 0.0)
            sat_speed = sat_speed_hi | sat_speed_lo
            tau_net[sat_speed] = 0.0
            tau_motor_eff[sat_speed] = tau_fric[sat_speed]
            omega_next = omega + dt_s * (tau_net / j_kg_m2)
            omega_next = np.clip(omega_next, -max_speed_rad_s, max_speed_rad_s)
            h_next = np.clip(j_kg_m2 * omega_next, -max_momentum_nms, max_momentum_nms)
            omega_next = h_next / j_kg_m2

            # Clipping can reduce the actual momentum change below the requested
            # net wheel torque.  Couple body torque to the realized momentum delta
            # so saturation cannot create angular impulse from nowhere.
            realized_tau_net = (
                (h_next - h_now) / dt_s if dt_s > 0.0 else tau_net.copy()
            )

            self.wheel_motor_torque_nm = tau_motor_eff
            self.wheel_speed_rad_s = omega_next
            self.wheel_momentum_wheels_nms = h_next
            self.wheel_momentum_nms = g @ h_next

            torque_body_nm = -(g @ realized_tau_net)
        diag = {
            "rw_num_wheels": int(n_wheels),
            "rw_torque_cmd_nm": tau_cmd.tolist(),
            "rw_motor_torque_nm": tau_motor_eff.tolist(),
            "rw_torque_applied_nm": tau_motor_eff.tolist(),
            "rw_net_wheel_torque_nm": realized_tau_net.tolist(),
            "rw_body_torque_applied_nm": torque_body_nm.tolist(),
            "rw_speed_rad_s": omega_next.tolist(),
            "rw_momentum_wheels_nms": h_next.tolist(),
            "rw_momentum_body_nms": self.wheel_momentum_nms.tolist(),
        }
        if backend == "rust":
            diag["rw_numeric_backend"] = backend
        return torque_body_nm, diag

    def _apply_magnetorquers(self, torque_body_cmd_nm: np.ndarray, mode_flags: dict) -> tuple[np.ndarray, dict]:
        mt = self.magnetorquers
        if mt is None:
            return torque_body_cmd_nm, {}
        max_dipole = np.abs(np.array(mt.max_dipole_a_m2, dtype=float).reshape(-1))
        if max_dipole.size == 1:
            max_dipole = np.full(3, float(max_dipole[0]), dtype=float)
        if max_dipole.size != 3:
            raise ValueError("max_dipole_a_m2 must be scalar or length-3.")

        b_body = mode_flags.get("magnetic_field_body_t")
        if b_body is None and mode_flags.get("magnetic_field_eci_t") is not None:
            q = mode_flags.get("current_attitude_quat_bn")
            if q is not None:
                c_bn = quaternion_to_dcm_bn(np.array(q, dtype=float).reshape(4))
                b_body = c_bn @ np.array(mode_flags["magnetic_field_eci_t"], dtype=float).reshape(3)
        if b_body is None:
            return np.zeros(3, dtype=float), {
                "magnetorquer_mode": "no_b_field_zero_torque",
                "magnetorquer_dipole_cmd_a_m2": np.zeros(3, dtype=float).tolist(),
                "magnetorquer_torque_body_nm": np.zeros(3, dtype=float).tolist(),
            }

        b = np.array(b_body, dtype=float).reshape(3)
        b2 = float(np.dot(b, b))
        if b2 <= 1e-24:
            dipole = np.zeros(3, dtype=float)
            torque = np.zeros(3, dtype=float)
        else:
            desired = np.array(torque_body_cmd_nm, dtype=float).reshape(3)
            dipole = np.cross(b, desired) / b2
            dipole = np.clip(dipole, -max_dipole, max_dipole)
            torque = np.cross(dipole, b)
        return torque, {
            "magnetorquer_mode": "physical_b_cross_m",
            "magnetic_field_body_t": b.tolist(),
            "magnetorquer_dipole_cmd_a_m2": dipole.tolist(),
            "magnetorquer_torque_body_nm": torque.tolist(),
        }

    def _apply_control_moment_gyros(self, torque_body_cmd_nm: np.ndarray, *, dt_s: float) -> tuple[np.ndarray, dict]:
        cmg = self.control_moment_gyros
        if cmg is None:
            return torque_body_cmd_nm, {}
        max_torque = self._as_vector(cmg.max_torque_nm, 3, default=0.0)
        momentum = self._as_vector(cmg.momentum_nms, 3, default=0.0)
        rate_limit = self._as_vector(cmg.gimbal_rate_limit_rad_s, 3, default=np.inf)
        gimbal_torque_cap = np.zeros(3, dtype=float)
        finite_rate = np.isfinite(rate_limit)
        gimbal_torque_cap[finite_rate] = np.abs(momentum[finite_rate] * rate_limit[finite_rate])
        gimbal_torque_cap[~finite_rate & (np.abs(momentum) > 0.0)] = np.inf
        cap = np.minimum(np.abs(max_torque), gimbal_torque_cap)
        target = np.clip(np.array(torque_body_cmd_nm, dtype=float).reshape(3), -cap, cap)
        tau_s = float(max(cmg.torque_time_constant_s, 0.0))
        if dt_s > 0.0 and tau_s > 0.0:
            alpha = float(np.clip(dt_s / tau_s, 0.0, 1.0))
            target = self.cmg_torque_nm + alpha * (target - self.cmg_torque_nm)
        self.cmg_torque_nm = target.copy()
        return target, {
            "cmg_torque_body_nm": target.tolist(),
            "cmg_torque_cap_nm": cap.tolist(),
        }

    def _desaturation_torque(self, h_body_nms: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        desat = self.wheel_desaturation
        if desat is None or self.reaction_wheels is None:
            return np.zeros(3, dtype=float), {}
        h_body = (
            np.array(self.wheel_momentum_nms, dtype=float).reshape(3)
            if h_body_nms is None
            else np.array(h_body_nms, dtype=float).reshape(3)
        )
        h_norm = float(np.linalg.norm(h_body))
        max_h = np.array(self.reaction_wheels.max_momentum_nms, dtype=float).reshape(-1)
        threshold = float(np.clip(desat.momentum_fraction_threshold, 0.0, 1.0)) * float(np.linalg.norm(max_h))
        if h_norm <= threshold or h_norm <= 0.0:
            return np.zeros(3, dtype=float), {
                "wheel_desaturation_active": False,
                "wheel_desaturation_torque_body_nm": [0.0, 0.0, 0.0],
            }
        unload = -float(max(desat.unload_gain_s_inv, 0.0)) * h_body
        unload_norm = float(np.linalg.norm(unload))
        max_unload = float(max(desat.max_unload_torque_nm, 0.0))
        if max_unload <= 0.0:
            unload = np.zeros(3, dtype=float)
        if unload_norm > max_unload > 0.0:
            unload *= max_unload / unload_norm
        return unload, {
            "wheel_desaturation_active": True,
            "wheel_desaturation_torque_body_nm": unload.tolist(),
            "wheel_desaturation_momentum_norm_nms": h_norm,
        }
