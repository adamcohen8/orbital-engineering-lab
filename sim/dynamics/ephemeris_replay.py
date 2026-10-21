"""Explicit passive prescribed-trajectory dynamics over an imported PV arc.

This is interpolation of supplied evidence, not ONP/OGP propagation. It can
be installed on a SimObject through the supported Python engine interfaces.
"""

from __future__ import annotations

import numpy as np

from sim.core.interfaces import DynamicsModel
from sim.core.models import Command, StateTruth
from sim.data_io.contracts import DataIOError
from sim.data_io.trajectory import TrajectorySampler


class PrescribedTrajectoryDynamics(DynamicsModel):
    def __init__(self, sampler: TrajectorySampler, *, simulation_epoch):
        trajectory = sampler.trajectory
        if trajectory.center.upper() != "EARTH" or trajectory.frame != "EME2000":
            raise DataIOError("StateTruth replay requires explicitly converted Earth/EME2000 data.")
        self.sampler = sampler
        self.epoch_offset_s = simulation_epoch.tai_seconds - trajectory.epoch.tai_seconds

    def step(self, state: StateTruth, command: Command, env: dict, dt_s: float) -> StateTruth:
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise DataIOError("Replay timestep must be finite and positive.")
        if np.any(command.thrust_eci_km_s2) or np.any(command.torque_body_nm) or command.mode_flags:
            raise DataIOError("Prescribed replay is passive; control/resource commands are not supported.")
        expected = self.sampler.state_at(state.t_s + self.epoch_offset_s)
        if not np.allclose(np.r_[state.position_eci_km, state.velocity_eci_km_s], expected, atol=1e-8, rtol=0):
            raise DataIOError("Replay initial/current state does not match the prescribed trajectory.")
        pv = self.sampler.state_at(state.t_s + dt_s + self.epoch_offset_s)
        result = state.copy()
        result.position_eci_km, result.velocity_eci_km_s = pv[:3], pv[3:]
        result.t_s += dt_s
        return result
