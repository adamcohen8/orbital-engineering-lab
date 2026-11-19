"""Connect synchronized docking physics to two satellite FSW runtimes."""

from __future__ import annotations

import numpy as np

from sim.core.models import StateTruth
from sim.flight_software import DockingPortMeasurement
from sim.utils.quaternion import quaternion_to_dcm_bn


class DockingFlightSoftware:
    def __init__(self, agents, object_ids, masses):
        self.object_ids = object_ids
        self.masses = masses
        self.runtimes = tuple(agents[oid].flight_software_runtime for oid in object_ids)
        self.commands = ()
        self.capture_armed = False
        self.hard_mate_requested = False

    def truths(self, pair, time_s):
        return {
            oid: StateTruth(row[:3].copy(), row[3:6].copy(), row[6:10].copy(),
                            row[10:13].copy(), mass, time_s)
            for oid, row, mass in zip(self.object_ids, pair, self.masses)
        }

    def prepare(self, pair, geometry, phase, time_ns):
        world = self.truths(pair, time_ns / 1e9)
        # Sample both ports and the peer's previous arm indication before
        # either stack runs. Neither stack sees the other's same-tick output.
        peer_arms = [runtime.docking_request(time_ns).arm_capture for runtime in self.runtimes]
        for index, (oid, runtime) in enumerate(zip(self.object_ids, self.runtimes)):
            measurement = DockingPortMeasurement(
                tuple(geometry.offset_m), tuple(geometry.speed_m_s),
                tuple(geometry.rotation_error_rad), tuple(geometry.omega_eci[0] - geometry.omega_eci[1]),
                tuple(tuple(axis) for axis in geometry.axes[1].T),
                phase, peer_arms[1 - index],
            )
            runtime.prepare_interval(
                world[oid], start_time_ns=time_ns, world_truth=world, docking_measurement=measurement,
            )
        requests = [runtime.docking_request(time_ns) for runtime in self.runtimes]
        self.capture_armed = all(request.arm_capture for request in requests)
        self.hard_mate_requested = all(request.request_hard_mate for request in requests)
        return world

    def next_boundary(self, start_ns, stop_ns):
        return min(runtime.next_hard_boundary_ns(after_time_ns=start_ns, before_time_ns=stop_ns)
                   for runtime in self.runtimes)

    def realize(self, world, start_ns, end_ns):
        self.commands = tuple(runtime.command_interval(
            world[oid], start_time_ns=start_ns, end_time_ns=end_ns, world_truth=world,
        ) for oid, runtime in zip(self.object_ids, self.runtimes))
        if any(command.mass_flow_kg_s for command in self.commands):
            raise ValueError("Commanded docking requires fixed-mass actuation.")

    def wrenches(self, pair):
        axes = np.stack([quaternion_to_dcm_bn(row[6:10]).T for row in pair])
        forces = np.array([command.force_eci_n for command in self.commands])
        forces += np.einsum("ijk,ik->ij", axes, [cmd.force_body_n for cmd in self.commands])
        return forces / self.masses[:, None] / 1000., np.array([cmd.torque_body_n_m for cmd in self.commands])
