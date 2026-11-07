from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal, InvalidOperation, localcontext
from typing import Literal

import numpy as np

from sim.core.interfaces import Actuator
from sim.core.models import Command
from sim.numeric_backend import normalize_numeric_backend
from sim.utils.quaternion import quaternion_to_dcm_bn


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.array(vec, dtype=float).reshape(3)
    mag = float(np.linalg.norm(arr))
    if mag <= eps:
        return np.zeros(3, dtype=float)
    return arr / mag


def effective_max_accel_km_s2(
    *,
    current_mass_kg: float,
    max_accel_km_s2: float = 0.0,
    max_thrust_n: float | None = None,
) -> float:
    limits_km_s2: list[float] = []
    accel_cap = float(max(max_accel_km_s2, 0.0))
    if accel_cap > 0.0:
        limits_km_s2.append(accel_cap)
    if max_thrust_n is not None:
        thrust_cap_n = float(max(max_thrust_n, 0.0))
        if thrust_cap_n <= 0.0:
            return 0.0
        if current_mass_kg > 0.0 and np.isfinite(float(current_mass_kg)):
            limits_km_s2.append(thrust_cap_n / float(current_mass_kg) / 1e3)
    if not limits_km_s2:
        return 0.0
    return float(max(min(limits_km_s2), 0.0))


def attitude_coupled_thrust_eci(
    commanded_accel_eci_km_s2: np.ndarray,
    *,
    attitude_quat_bn: np.ndarray,
    thruster_direction_body: np.ndarray,
) -> np.ndarray:
    accel_cmd = np.array(commanded_accel_eci_km_s2, dtype=float).reshape(3)
    accel_mag = float(np.linalg.norm(accel_cmd))
    if accel_mag <= 0.0:
        return np.zeros(3, dtype=float)
    plume_axis_body = _unit(np.array(thruster_direction_body, dtype=float).reshape(3))
    if float(np.linalg.norm(plume_axis_body)) <= 0.0:
        return accel_cmd
    c_bn = quaternion_to_dcm_bn(np.array(attitude_quat_bn, dtype=float).reshape(4))
    plume_axis_eci = c_bn.T @ plume_axis_body
    # The stored mount axis is the nozzle / plume direction, so vehicle force is opposite it.
    return -accel_mag * plume_axis_eci


def thruster_disturbance_torque_body_nm(
    applied_accel_eci_km_s2: np.ndarray,
    *,
    current_mass_kg: float,
    thruster_direction_body: np.ndarray,
    thruster_position_body_m: np.ndarray,
) -> np.ndarray:
    accel_mag_m_s2 = float(np.linalg.norm(np.array(applied_accel_eci_km_s2, dtype=float).reshape(3)) * 1e3)
    if accel_mag_m_s2 <= 0.0 or current_mass_kg <= 0.0:
        return np.zeros(3, dtype=float)
    plume_axis_body = _unit(np.array(thruster_direction_body, dtype=float).reshape(3))
    if float(np.linalg.norm(plume_axis_body)) <= 0.0:
        return np.zeros(3, dtype=float)
    mount_position_body_m = np.array(thruster_position_body_m, dtype=float).reshape(3)
    force_body_n = -float(current_mass_kg) * accel_mag_m_s2 * plume_axis_body
    return np.cross(mount_position_body_m, force_body_n)


@dataclass(frozen=True)
class OrbitalActuatorLimits:
    max_accel_km_s2: float
    max_thrust_n: float | None = None
    min_impulse_bit_km_s: float = 0.0
    max_throttle_rate_km_s2_s: float = 1e-6
    isp_s: float = 220.0
    thruster_direction_body: np.ndarray | None = None
    thruster_position_body_m: np.ndarray | None = None
    couple_to_attitude: bool = True
    rcs_cluster: RcsClusterLimits | None = None
    electric_propulsion: ElectricPropulsionLimits | None = None
    gimbaled_thruster: GimbaledThrusterLimits | None = None


@dataclass(frozen=True)
class RcsThruster:
    name: str
    position_body_m: np.ndarray
    force_direction_body: np.ndarray
    max_thrust_n: float
    min_impulse_bit_n_s: float = 0.0
    isp_s: float = 220.0


@dataclass(frozen=True)
class RcsClusterLimits:
    thrusters: tuple[RcsThruster, ...]
    allocation_mode: Literal["force_torque", "torque_only", "force_only"] = "force_torque"
    pulse_quantum_s: float = 0.0
    duty_cycle: float = 1.0
    force_weight: float = 1.0
    torque_weight: float = 1.0
    numeric_backend: str = "rust"


@dataclass(frozen=True)
class ElectricPropulsionLimits:
    max_thrust_n: float
    isp_s: float = 1500.0
    duty_cycle: float = 1.0
    max_power_w: float | None = None
    power_per_newton_w: float | None = None
    throttle_time_constant_s: float = 0.0


@dataclass(frozen=True)
class GimbaledThrusterLimits:
    neutral_direction_body: np.ndarray
    position_body_m: np.ndarray | None = None
    max_gimbal_angle_rad: float = 0.0
    max_gimbal_rate_rad_s: float = np.inf
    response_time_constant_s: float = 0.0


@dataclass
class OrbitalActuator(Actuator):
    lag_tau_s: float = 0.0
    _last_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _last_electric_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _gimbal_direction_body: np.ndarray | None = None
    _rcs_staging_cache: tuple | None = field(default=None, init=False, repr=False)

    def apply(self, command: Command, limits: dict, dt_s: float) -> Command:
        try:
            dt_s = float(dt_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt_s must be finite and nonnegative") from exc
        if not np.isfinite(dt_s):
            raise ValueError("dt_s must be finite")
        if dt_s < 0.0:
            raise ValueError("dt_s must be nonnegative")

        lim: OrbitalActuatorLimits = limits["orbital"]
        accel_filtered = np.array(command.thrust_eci_km_s2, dtype=float)
        torque_applied = np.array(command.torque_body_nm, dtype=float)
        mode_flags = dict(command.mode_flags)
        thruster_direction_body = mode_flags.get("thruster_direction_body", lim.thruster_direction_body)
        thruster_position_body_m = mode_flags.get("thruster_position_body_m", lim.thruster_position_body_m)
        current_mass_kg = float(mode_flags.get("current_mass_kg", mode_flags.get("mass_kg", 0.0)))
        effective_max_accel = effective_max_accel_km_s2(
            current_mass_kg=current_mass_kg,
            max_accel_km_s2=lim.max_accel_km_s2,
            max_thrust_n=lim.max_thrust_n,
        )
        attitude_quat_bn = mode_flags.get("current_attitude_quat_bn")
        mass_isp_s = float(lim.isp_s)

        if lim.rcs_cluster is not None:
            return self._apply_rcs_cluster(
                command=command,
                cluster=lim.rcs_cluster,
                mode_flags=mode_flags,
                dt_s=dt_s,
                current_mass_kg=current_mass_kg,
                attitude_quat_bn=attitude_quat_bn,
            )

        if lim.electric_propulsion is not None:
            accel_filtered, electric_diag = self._apply_electric_propulsion(
                accel_filtered,
                ep=lim.electric_propulsion,
                current_mass_kg=current_mass_kg,
                dt_s=dt_s,
            )
            mode_flags.update(electric_diag)
            mass_isp_s = float(lim.electric_propulsion.isp_s)

        norm = np.linalg.norm(accel_filtered)
        if norm > effective_max_accel > 0.0:
            accel_filtered *= effective_max_accel / norm
        elif effective_max_accel == 0.0:
            accel_filtered = np.zeros(3, dtype=float)

        max_delta = lim.max_throttle_rate_km_s2_s * dt_s
        delta = accel_filtered - self._last_accel
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta > 0.0:
            accel_filtered = self._last_accel + delta * (max_delta / delta_norm)

        if self.lag_tau_s > 0.0:
            alpha = min(1.0, dt_s / self.lag_tau_s)
            accel_filtered = self._last_accel + alpha * (accel_filtered - self._last_accel)

        dv = float(np.linalg.norm(accel_filtered) * dt_s)
        if 0.0 < dv < lim.min_impulse_bit_km_s:
            accel_filtered = np.zeros(3)

        self._last_accel = accel_filtered.copy()
        accel_applied = accel_filtered.copy()
        gimbaled_direction_body = None
        if bool(lim.couple_to_attitude) and lim.gimbaled_thruster is not None and attitude_quat_bn is not None:
            accel_applied, gimbaled_direction_body, gimbal_diag = self._apply_gimbaled_thruster(
                accel_filtered,
                gimbal=lim.gimbaled_thruster,
                attitude_quat_bn=np.array(attitude_quat_bn, dtype=float),
                dt_s=dt_s,
            )
            mode_flags.update(gimbal_diag)
            thruster_direction_body = gimbaled_direction_body
            if lim.gimbaled_thruster.position_body_m is not None:
                thruster_position_body_m = lim.gimbaled_thruster.position_body_m
        elif bool(lim.couple_to_attitude):
            if thruster_direction_body is not None and attitude_quat_bn is not None:
                accel_applied = attitude_coupled_thrust_eci(
                    accel_filtered,
                    attitude_quat_bn=np.array(attitude_quat_bn, dtype=float),
                    thruster_direction_body=np.array(thruster_direction_body, dtype=float),
                )
        g0_m_s2 = 9.80665
        accel_mag_m_s2 = float(np.linalg.norm(accel_applied) * 1e3)
        thrust_n = max(current_mass_kg, 0.0) * accel_mag_m_s2
        mdot_kg_s = 0.0 if mass_isp_s <= 0.0 or thrust_n <= 0.0 else thrust_n / (mass_isp_s * g0_m_s2)
        mode_flags["delta_mass_kg"] = float(mdot_kg_s * dt_s)
        mode_flags["effective_max_accel_km_s2"] = float(effective_max_accel)
        if lim.max_thrust_n is not None:
            mode_flags["max_thrust_n"] = float(max(lim.max_thrust_n, 0.0))
        if gimbaled_direction_body is not None:
            mode_flags["thruster_direction_body"] = np.array(gimbaled_direction_body, dtype=float).tolist()
        thruster_torque_body_nm = np.zeros(3, dtype=float)
        if thruster_direction_body is not None and thruster_position_body_m is not None:
            thruster_torque_body_nm = thruster_disturbance_torque_body_nm(
                accel_applied,
                current_mass_kg=current_mass_kg,
                thruster_direction_body=np.array(thruster_direction_body, dtype=float),
                thruster_position_body_m=np.array(thruster_position_body_m, dtype=float),
            )
            torque_applied = torque_applied + thruster_torque_body_nm
        mode_flags["thruster_torque_body_nm"] = thruster_torque_body_nm.tolist()
        return Command(thrust_eci_km_s2=accel_applied, torque_body_nm=torque_applied, mode_flags=mode_flags)

    def _apply_electric_propulsion(
        self,
        accel_cmd_eci_km_s2: np.ndarray,
        *,
        ep: ElectricPropulsionLimits,
        current_mass_kg: float,
        dt_s: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        accel = np.array(accel_cmd_eci_km_s2, dtype=float).reshape(3)
        max_thrust_n = float(max(ep.max_thrust_n, 0.0))
        if ep.max_power_w is not None and ep.power_per_newton_w is not None and ep.power_per_newton_w > 0.0:
            max_thrust_n = min(max_thrust_n, float(max(ep.max_power_w, 0.0)) / float(ep.power_per_newton_w))
        max_thrust_n *= float(np.clip(ep.duty_cycle, 0.0, 1.0))
        max_accel = 0.0 if current_mass_kg <= 0.0 else max_thrust_n / current_mass_kg / 1e3
        norm = float(np.linalg.norm(accel))
        if norm > max_accel > 0.0:
            accel = accel * (max_accel / norm)
        elif max_accel <= 0.0:
            accel = np.zeros(3, dtype=float)
        tau_s = float(max(ep.throttle_time_constant_s, 0.0))
        if dt_s > 0.0 and tau_s > 0.0:
            alpha = float(np.clip(dt_s / tau_s, 0.0, 1.0))
            accel = self._last_electric_accel + alpha * (accel - self._last_electric_accel)
        self._last_electric_accel = accel.copy()
        g0_m_s2 = 9.80665
        thrust_n = max(current_mass_kg, 0.0) * float(np.linalg.norm(accel) * 1e3)
        mdot_kg_s = 0.0 if ep.isp_s <= 0.0 or thrust_n <= 0.0 else thrust_n / (float(ep.isp_s) * g0_m_s2)
        return accel, {
            "electric_propulsion_thrust_n": float(thrust_n),
            "electric_propulsion_max_thrust_n": float(max_thrust_n),
            "electric_propulsion_delta_mass_kg": float(mdot_kg_s * max(dt_s, 0.0)),
        }

    def _apply_gimbaled_thruster(
        self,
        accel_cmd_eci_km_s2: np.ndarray,
        *,
        gimbal: GimbaledThrusterLimits,
        attitude_quat_bn: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        accel = np.array(accel_cmd_eci_km_s2, dtype=float).reshape(3)
        accel_mag = float(np.linalg.norm(accel))
        neutral = _unit(np.array(gimbal.neutral_direction_body, dtype=float).reshape(3))
        if accel_mag <= 0.0 or float(np.linalg.norm(neutral)) <= 0.0:
            if self._gimbal_direction_body is None:
                self._gimbal_direction_body = neutral.copy()
            return np.zeros(3, dtype=float), self._gimbal_direction_body.copy(), {
                "gimbal_angle_rad": 0.0,
                "gimbal_rate_limited": False,
            }
        c_bn = quaternion_to_dcm_bn(np.array(attitude_quat_bn, dtype=float).reshape(4))
        desired_force_body = _unit(c_bn @ accel)
        desired_plume_body = -desired_force_body
        limited_target = _rotate_toward(neutral, desired_plume_body, float(max(gimbal.max_gimbal_angle_rad, 0.0)))
        current = neutral if self._gimbal_direction_body is None else _unit(self._gimbal_direction_body)
        max_step = float(max(gimbal.max_gimbal_rate_rad_s, 0.0)) * max(dt_s, 0.0)
        if gimbal.response_time_constant_s > 0.0:
            max_step = min(max_step, float(max(dt_s, 0.0) / gimbal.response_time_constant_s))
        rate_limited = _angle_between(current, limited_target) > max_step + 1e-12
        next_dir = _rotate_toward(current, limited_target, max_step)
        self._gimbal_direction_body = next_dir.copy()
        achieved_force_eci = c_bn.T @ (-next_dir)
        gimbal_angle = _angle_between(neutral, next_dir)
        return accel_mag * achieved_force_eci, next_dir, {
            "gimbal_direction_body": next_dir.tolist(),
            "gimbal_angle_rad": float(gimbal_angle),
            "gimbal_rate_limited": bool(rate_limited),
        }

    def _apply_rcs_cluster(
        self,
        *,
        command: Command,
        cluster: RcsClusterLimits,
        mode_flags: dict,
        dt_s: float,
        current_mass_kg: float,
        attitude_quat_bn: np.ndarray | None,
    ) -> Command:
        thrusters = tuple(cluster.thrusters or ())
        if not thrusters or current_mass_kg <= 0.0:
            mode_flags["rcs_thruster_forces_n"] = []
            return Command(
                thrust_eci_km_s2=np.zeros(3, dtype=float),
                torque_body_nm=np.zeros(3, dtype=float),
                mode_flags=mode_flags,
            )
        c_bn = (
            quaternion_to_dcm_bn(np.array(attitude_quat_bn, dtype=float).reshape(4))
            if attitude_quat_bn is not None
            else np.eye(3)
        )
        desired_force_eci_n = np.array(command.thrust_eci_km_s2, dtype=float).reshape(3) * current_mass_kg * 1e3
        desired_force_body_n = c_bn @ desired_force_eci_n
        desired_torque_body_nm = np.array(command.torque_body_nm, dtype=float).reshape(3)
        if cluster.allocation_mode == "torque_only":
            target = desired_torque_body_nm
            rows = slice(3, 6)
        elif cluster.allocation_mode == "force_only":
            target = desired_force_body_n
            rows = slice(0, 3)
        else:
            target = np.hstack((desired_force_body_n, desired_torque_body_nm))
            rows = slice(0, 6)

        # Frozen device records may contain mutable arrays. Bind staging to
        # their contents so edits still affect the very next command.
        signature = (
            cluster.allocation_mode, float(cluster.force_weight), float(cluster.torque_weight),
            tuple((np.asarray(item.force_direction_body, dtype=float).tobytes(),
                   np.asarray(item.position_body_m, dtype=float).tobytes(),
                   float(item.max_thrust_n), float(item.min_impulse_bit_n_s),
                   float(item.isp_s), str(item.name)) for item in thrusters),
        )
        cached = self._rcs_staging_cache
        if cached is None or cached[0] != signature:
            force_dirs = []
            torque_dirs = []
            max_forces = []
            min_impulses = []
            isps = []
            names = []
            for thruster in thrusters:
                force_dir = _unit(np.array(thruster.force_direction_body, dtype=float).reshape(3))
                pos = np.array(thruster.position_body_m, dtype=float).reshape(3)
                force_dirs.append(force_dir)
                torque_dirs.append(np.cross(pos, force_dir))
                max_forces.append(float(max(thruster.max_thrust_n, 0.0)))
                min_impulses.append(float(max(thruster.min_impulse_bit_n_s, 0.0)))
                isps.append(float(thruster.isp_s))
                names.append(str(thruster.name))
            allocation = np.vstack((np.column_stack(force_dirs), np.column_stack(torque_dirs)))[rows, :]
            solve_allocation = np.asarray(allocation, dtype=float)
            row_scale = None
            if cluster.allocation_mode == "force_torque":
                force_scale = max(float(np.sum(max_forces)), 1e-12)
                torque_capacity = sum(
                    max_force * float(np.linalg.norm(torque_axis))
                    for max_force, torque_axis in zip(max_forces, torque_dirs, strict=True)
                )
                torque_scale = max(float(torque_capacity), 1e-12)
                row_scale = np.hstack(
                    (
                        np.full(3, float(max(cluster.force_weight, 0.0)) / force_scale),
                        np.full(3, float(max(cluster.torque_weight, 0.0)) / torque_scale),
                    )
                )
                solve_allocation = solve_allocation * row_scale[:, None]
            force_matrix = np.array(force_dirs, dtype=float).T
            torque_matrix = np.array(torque_dirs, dtype=float).T
            cached = (signature, solve_allocation, row_scale, force_matrix, torque_matrix,
                      max_forces, min_impulses, isps, names)
            self._rcs_staging_cache = cached
        _, solve_allocation, row_scale, force_matrix, torque_matrix, max_forces, min_impulses, isps, names = cached
        solve_target = np.array(target, dtype=float).reshape(-1)
        if row_scale is not None:
            solve_target = solve_target * row_scale
        backend = normalize_numeric_backend(cluster.numeric_backend, error_message="RCS numeric_backend must be python or rust")
        # Preserve SciPy's trust-region reflective solve and its diagnostics.
        # A greedy active-set approximation changes redundant and rank-deficient
        # RCS allocations, so this path intentionally remains Python even when
        # the requested numeric backend is Rust.
        forces = _bounded_nonnegative_lstsq(solve_allocation, solve_target, np.array(max_forces))
        try:
            duty_raw = float(cluster.duty_cycle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("RCS duty_cycle must be finite") from exc
        if not np.isfinite(duty_raw):
            raise ValueError("RCS duty_cycle must be finite")
        duty = float(np.clip(duty_raw, 0.0, 1.0))
        forces *= duty
        try:
            quantum_s = float(cluster.pulse_quantum_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("RCS pulse_quantum_s must be finite and nonnegative") from exc
        if not np.isfinite(quantum_s):
            raise ValueError("RCS pulse_quantum_s must be finite")
        if quantum_s < 0.0:
            raise ValueError("RCS pulse_quantum_s must be nonnegative")
        if quantum_s > 0.0 and dt_s > 0.0:
            requested_on_time_s = (forces / np.maximum(max_forces, 1e-12)) * dt_s
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                requested_pulses_real = requested_on_time_s / quantum_s
            if not np.all(np.isfinite(requested_pulses_real)):
                raise ValueError("RCS pulse quantization ratio is not representable")
            requested_pulses = np.rint(requested_pulses_real)
            # Duty cycle is an availability cap, so pulse quantization must
            # never round the realized on-time back above the duty-limited
            # value. Keep the result on the pulse grid as well: if the duty
            # window is shorter than one quantum, no pulse is feasible.
            # Use the short decimal spellings of the finite inputs for the
            # boundary calculation. This recovers an intended 0.3 / 0.1 = 3
            # pulse boundary without adding a scale-dependent ULP allowance
            # that could admit many pulses at large, finite ratios.
            try:
                with localcontext() as context:
                    context.prec = 80
                    max_pulses_decimal = (
                        Decimal(str(dt_s)) * Decimal(str(duty)) / Decimal(str(quantum_s))
                    )
                    max_pulses_decimal = max_pulses_decimal.to_integral_value(rounding=ROUND_FLOOR)
                max_pulses = float(max_pulses_decimal)
            except (InvalidOperation, OverflowError, ValueError) as exc:
                raise ValueError("RCS pulse quantization ratio is not representable") from exc
            if not np.isfinite(max_pulses):
                raise ValueError("RCS pulse quantization ratio is not representable")
            # A float that rounded the integer cap upward could admit a pulse
            # beyond the duty window. If the integer pulse count itself is not
            # exactly representable, fail closed rather than inventing a
            # scale-dependent scalar cutoff or silently violating the cap.
            if Decimal.from_float(max_pulses) != max_pulses_decimal:
                raise ValueError("RCS pulse quantization ratio is not representable")
            pulse_count = np.clip(requested_pulses, 0.0, max_pulses)
            on_time = pulse_count * quantum_s
            if not np.all(np.isfinite(on_time)):
                raise ValueError("RCS pulse quantization ratio is not representable")
            realized_fraction = on_time / dt_s
            forces = np.array(max_forces, dtype=float) * realized_fraction
            # Multiplication/division can place a mathematically exact pulse
            # boundary one float above the duty cap (e.g. 3 * 0.1 > 0.3).
            # Clamp only that final representation to the cap; pulse_count and
            # on_time remain whole-quantum values.
            forces = np.minimum(forces, np.array(max_forces, dtype=float) * duty)
        if dt_s > 0.0:
            for idx, min_impulse in enumerate(min_impulses):
                if 0.0 < forces[idx] * dt_s < min_impulse:
                    forces[idx] = 0.0
        if backend == "rust":
            # SciPy's bounded TRF solve, duty-cycle quantization, and minimum
            # impulse policy remain authoritative.  Native matvecs cover the
            # repeated achieved-force/torque products after that solve.
            from sim.rust_control_backend import mat_vec

            force_body_n = mat_vec(force_matrix, forces)
            rcs_torque_body_nm = mat_vec(torque_matrix, forces)
        else:
            force_body_n = np.sum(force_matrix * forces.reshape(1, -1), axis=1)
            rcs_torque_body_nm = np.sum(torque_matrix * forces.reshape(1, -1), axis=1)
        torque_body_nm = rcs_torque_body_nm
        if backend == "rust":
            force_eci_n = mat_vec(c_bn.T, force_body_n)
        else:
            force_eci_n = c_bn.T @ force_body_n
        accel_eci_km_s2 = force_eci_n / current_mass_kg / 1e3
        g0_m_s2 = 9.80665
        mdot = 0.0
        for force_n, isp_s in zip(forces, isps):
            if force_n > 0.0 and isp_s > 0.0:
                mdot += float(force_n / (isp_s * g0_m_s2))
        mode_flags["rcs_thruster_names"] = list(names)
        mode_flags["rcs_thruster_forces_n"] = forces.tolist()
        mode_flags["rcs_thruster_max_forces_n"] = list(max_forces)
        mode_flags["rcs_allocation_saturated"] = bool(
            any(
                max_force > 0.0 and force >= max_force * duty - 1.0e-12
                for force, max_force in zip(forces, max_forces, strict=True)
            )
        )
        mode_flags["rcs_min_thrust_margin_n"] = float(
            min((max_force * duty - force for force, max_force in zip(forces, max_forces, strict=True)), default=0.0)
        )
        mode_flags["rcs_force_body_n"] = force_body_n.tolist()
        mode_flags["rcs_torque_body_nm"] = rcs_torque_body_nm.tolist()
        mode_flags["rcs_force_residual_n"] = (desired_force_body_n - force_body_n).tolist()
        mode_flags["rcs_torque_residual_nm"] = (desired_torque_body_nm - rcs_torque_body_nm).tolist()
        if backend == "rust":
            mode_flags["rcs_numeric_backend"] = backend
            mode_flags["rcs_numeric_path"] = "python_reference_trf_rust_matvec"
        mode_flags["delta_mass_kg"] = float(mdot * max(dt_s, 0.0))
        return Command(thrust_eci_km_s2=accel_eci_km_s2, torque_body_nm=torque_body_nm, mode_flags=mode_flags)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(a)
    ub = _unit(b)
    if float(np.linalg.norm(ua)) <= 0.0 or float(np.linalg.norm(ub)) <= 0.0:
        return 0.0
    return float(np.arccos(np.clip(float(np.dot(ua, ub)), -1.0, 1.0)))


def _rotate_toward(current: np.ndarray, target: np.ndarray, max_angle_rad: float) -> np.ndarray:
    cur = _unit(current)
    tgt = _unit(target)
    if float(np.linalg.norm(cur)) <= 0.0:
        return tgt
    if float(np.linalg.norm(tgt)) <= 0.0:
        return cur
    angle = _angle_between(cur, tgt)
    if angle <= max_angle_rad or angle <= 1e-12:
        return tgt
    axis = np.cross(cur, tgt)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(cur, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = np.cross(cur, helper)
        axis_norm = float(np.linalg.norm(axis))
    axis = axis / max(axis_norm, 1e-12)
    step = float(max(max_angle_rad, 0.0))
    out = cur * np.cos(step) + np.cross(axis, cur) * np.sin(step) + axis * np.dot(axis, cur) * (1.0 - np.cos(step))
    return _unit(out)


def _bounded_nonnegative_lstsq(a: np.ndarray, b: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # SciPy's optimizer imports a large solver stack. Most spacecraft never
    # use RCS allocation, so load it only for the bounded allocation path.
    from scipy.optimize import lsq_linear

    matrix = np.array(a, dtype=float)
    target = np.array(b, dtype=float).reshape(matrix.shape[0])
    upper = np.array(upper, dtype=float).reshape(matrix.shape[1])
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(target)) and np.all(np.isfinite(upper))):
        raise ValueError("RCS allocation inputs must be finite.")
    if np.any(upper < 0.0):
        raise ValueError("RCS allocation upper bounds must be nonnegative.")
    result = lsq_linear(
        matrix,
        target,
        bounds=(np.zeros(matrix.shape[1], dtype=float), upper),
        method="trf",
        tol=1e-12,
        lsmr_tol=1e-12,
        max_iter=max(100, 10 * matrix.shape[1]),
    )
    if not bool(result.success):
        raise RuntimeError(f"RCS bounded least-squares allocation failed: {result.message}")
    return np.clip(np.asarray(result.x, dtype=float), 0.0, upper)
