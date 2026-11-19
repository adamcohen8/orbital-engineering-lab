"""Six-DOF contact and rigid mating for a cooperative ring-and-face port pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from sim.dynamics.attitude.rigid_body import rigid_body_derivatives
from sim.dynamics.orbit.integrators import rk4_step_state
from sim.utils.quaternion import dcm_to_quaternion_bn, quaternion_to_dcm_bn

# The mating pose aligns port X and reverses the outward Y and Z axes.
_MATING_ROTATION = np.diag([1., -1., -1.])
# Four equally spaced, lumped contact sites on the probe ring.
_RING = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])


@dataclass(frozen=True)
class DockingPort:
    offset_m: np.ndarray
    axes_body: np.ndarray
    radius_m: float

    @classmethod
    def from_config(cls, config):
        return cls(
            np.array(config["offset_body_m"]),
            quaternion_to_dcm_bn(config["quat_port_from_body"]).T,
            config["radius_m"],
        )


@dataclass(frozen=True)
class PortGeometry:
    body_axes: np.ndarray
    axes: np.ndarray
    lever_m: np.ndarray
    omega_eci: np.ndarray
    offset_m: np.ndarray
    speed_m_s: np.ndarray
    lever_to_contact_m: np.ndarray
    gaps_m: np.ndarray
    gap_rates_m_s: np.ndarray
    radial_clearance_m: np.ndarray
    facing: bool

    @property
    def clearance_m(self) -> float:
        if not self.facing:
            return float("inf")
        return float(np.min(np.maximum(self.gaps_m, self.radial_clearance_m)))

    @property
    def lateral_m(self) -> float:
        return float(np.linalg.norm(self.offset_m[:2]))

    @property
    def rotation_error_rad(self) -> np.ndarray:
        expected = self.axes[1] @ _MATING_ROTATION
        return Rotation.from_matrix(self.axes[0] @ expected.T).as_rotvec()


class DockingInterface:
    def __init__(self, config):
        self.ports = tuple(DockingPort.from_config(config[role]) for role in ("probe", "receiver"))
        self.port_axes_body = np.stack([port.axes_body for port in self.ports])
        self.port_offsets_body_m = np.stack([port.offset_m for port in self.ports])
        self.ring_points_m = self.ports[0].radius_m * _RING
        self.stiffness = config["contact_stiffness_n_m"]
        self.damping = config["contact_damping_n_s_m"]
        self.latch = config.get("latch")

    def geometry(self, pair: np.ndarray) -> PortGeometry:
        body_axes = np.stack([quaternion_to_dcm_bn(row[6:10]).T for row in pair])
        axes = body_axes @ self.port_axes_body
        lever = np.einsum("ijk,ik->ij", body_axes, self.port_offsets_body_m)
        omega = np.einsum("ijk,ik->ij", body_axes, pair[:, 10:13])
        # Form relative differences before adding meter-scale offsets to ECI positions.
        delta_m = (pair[0, :3] - pair[1, :3]) * 1000.
        delta_v = (pair[0, 3:6] - pair[1, 3:6]) * 1000.
        offset = delta_m + lever[0] - lever[1]
        speed = delta_v + np.cross(omega[0], lever[0]) - np.cross(omega[1], lever[1])
        ring = self.ring_points_m @ axes[0].T
        in_receiver = (offset + ring) @ axes[1]
        normal = axes[1, :, 2]
        # Both lever arms refer to the same spatial point so the wrench is internal.
        contact = np.stack([lever[0] + ring, delta_m + lever[0] + ring])
        relative_velocity = delta_v + np.cross(omega[0], contact[0]) - np.cross(omega[1], contact[1])
        return PortGeometry(
            body_axes, axes, lever, omega, axes[1].T @ offset, axes[1].T @ speed, contact,
            in_receiver[:, 2], relative_velocity @ normal,
            np.linalg.norm(in_receiver[:, :2], axis=1) - self.ports[1].radius_m,
            bool(np.dot(axes[0, :, 2], normal) < 0.),
        )

    def validate_initial_geometry(self, pair: np.ndarray) -> None:
        geometry = self.geometry(pair)
        if not geometry.facing or geometry.offset_m[2] <= 0. or geometry.clearance_m <= 0.:
            raise ValueError("simulator.docking requires opposed ports with initially separated contact surfaces.")

    def can_capture(self, geometry: PortGeometry) -> bool:
        latch = self.latch
        return (latch is not None
                and np.linalg.norm(geometry.offset_m) <= latch["max_travel_m"]
                and geometry.lateral_m <= latch["capture_lateral_m"]
                and np.linalg.norm(geometry.rotation_error_rad) <= latch["capture_angle_rad"]
                and np.linalg.norm(geometry.speed_m_s) <= latch["capture_speed_m_s"]
                and np.linalg.norm(geometry.omega_eci[0] - geometry.omega_eci[1]) <= latch["capture_rate_rad_s"])

    def wrenches(self, pair: np.ndarray, geometry: PortGeometry, *, captured: bool):
        normal = geometry.axes[1, :, 2]
        contact_mask = geometry.facing & (geometry.radial_clearance_m <= 0.) & (geometry.gaps_m <= 0.)
        loads_n = np.where(contact_mask, np.maximum(
            0., -self.stiffness * geometry.gaps_m - self.damping * geometry.gap_rates_m_s,
        ), 0.)
        forces = loads_n[:, None] * normal
        torque = np.stack([np.cross(lever, forces).sum(axis=0) for lever in geometry.lever_to_contact_m])
        force = forces.sum(axis=0)
        latch_force = np.zeros(3)
        latch_torque = np.zeros(3)
        if captured:
            latch = self.latch
            delta = geometry.axes[1] @ geometry.offset_m
            midpoint_lever = geometry.lever_m + np.stack([-0.5 * delta, 0.5 * delta])
            midpoint_speed = ((pair[0, 3:6] - pair[1, 3:6]) * 1000.
                              + np.cross(geometry.omega_eci[0], midpoint_lever[0])
                              - np.cross(geometry.omega_eci[1], midpoint_lever[1]))
            latch_force = -latch["stiffness_n_m"] * delta - latch["damping_n_s_m"] * midpoint_speed
            latch_torque = (-latch["angular_stiffness_n_m_rad"] * geometry.rotation_error_rad
                            - latch["angular_damping_n_m_s_rad"] * (geometry.omega_eci[0] - geometry.omega_eci[1]))
            torque += np.cross(midpoint_lever, latch_force) + latch_torque
            force += latch_force
        return np.stack([force, -force]), torque * np.array([[1.], [-1.]]), loads_n, latch_force, latch_torque


@dataclass(frozen=True)
class MatedAssembly:
    """Rigidly constrain the as-latched poses, preserving both object identities."""

    offsets_m: np.ndarray
    body_axes: np.ndarray
    inertia_kg_m2: np.ndarray
    masses_kg: np.ndarray

    @classmethod
    def lock(
        cls, pair: np.ndarray, masses: np.ndarray, inertias: np.ndarray,
    ) -> tuple[MatedAssembly, np.ndarray]:
        total = float(np.sum(masses))
        relative_m = (pair[0, :3] - pair[1, :3]) * 1000.
        offsets_eci = np.stack([relative_m * (masses[1] / total), -relative_m * (masses[0] / total)])
        com = pair[1, :3] - offsets_eci[1] / 1000.
        velocity = (masses @ pair[:, 3:6]) / total
        axes = np.stack([quaternion_to_dcm_bn(row[6:10]).T for row in pair])
        reference = axes[1]
        body_axes = reference.T @ axes
        spin_inertia = axes @ inertias @ axes.transpose(0, 2, 1)
        total_inertia = sum(spin_inertia[i] + masses[i] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        ) for i, offset in enumerate(offsets_eci))
        momentum = sum(axes[i] @ inertias[i] @ pair[i, 10:13] + np.cross(
            offset, masses[i] * (pair[i, 3:6] - velocity) * 1000.,
        ) for i, offset in enumerate(offsets_eci))
        assembly = cls(offsets_eci @ reference, body_axes, reference.T @ total_inertia @ reference, masses)
        omega = np.linalg.solve(assembly.inertia_kg_m2, reference.T @ momentum)
        return assembly, np.concatenate([com, velocity, pair[1, 6:10], omega])

    def pair_state(self, state: np.ndarray) -> np.ndarray:
        axes = quaternion_to_dcm_bn(state[6:10]).T
        offsets = self.offsets_m @ axes.T
        omega = axes @ state[10:13]
        return np.column_stack([
            state[:3] + offsets / 1000.,
            state[3:6] + np.cross(omega, offsets) / 1000.,
            np.stack([dcm_to_quaternion_bn((axes @ body).T) for body in self.body_axes]),
            self.body_axes.transpose(0, 2, 1) @ state[10:13],
        ])

    def step(
        self, state: np.ndarray, time_s: float, width_s: float,
        gravity: Callable[[float, np.ndarray], np.ndarray],
        actuation: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        def derivative(t, stage):
            axes = quaternion_to_dcm_bn(stage[6:10]).T
            pair = self.pair_state(stage)
            acceleration = gravity(t, pair)
            if actuation is not None:
                applied_acceleration, body_torques = actuation(pair)
                acceleration += applied_acceleration
            net_acceleration = (self.masses_kg @ acceleration) / self.masses_kg.sum()
            differential_force_n = self.masses_kg[:, None] * (acceleration - net_acceleration) * 1000.
            torque = np.cross(self.offsets_m, differential_force_n @ axes).sum(axis=0)
            if actuation is not None:
                torque += np.einsum("ijk,ik->j", self.body_axes, body_torques)
            q_dot, w_dot = rigid_body_derivatives(
                stage[6:10], stage[10:13], self.inertia_kg_m2, torque, numeric_backend="python",
            )
            return np.concatenate([stage[3:6], net_acceleration, q_dot, w_dot])

        return rk4_step_state(derivative, time_s, state, width_s)
