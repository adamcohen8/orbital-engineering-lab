"""Owner-side synchronized ONP stepping and docking event evidence."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.attitude.rigid_body import rigid_body_derivatives
from sim.dynamics.docking import DockingInterface, MatedAssembly
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.integrators import rk4_step_state
from sim.execution.object_workers import ObjectStepInput, ObjectStepResult
from sim.runtime.models import AgentRuntime
from sim.utils.quaternion import normalize_quaternion

_EVENT_TOLERANCE_S = 1e-8


class DockingStepper:
    def __init__(
        self, config: dict[str, Any], agents: dict[str, AgentRuntime],
        base_environment: dict, max_step_s: float,
    ) -> None:
        self.config = config
        self.interface = DockingInterface(config)
        self.object_ids = tuple(config[role]["object_id"] for role in ("probe", "receiver"))
        self.dynamics = [agents[oid].dynamics for oid in self.object_ids]
        self.masses = np.array([agents[oid].truth.mass_kg for oid in self.object_ids])
        self.inertias = np.stack([dynamics.inertia_kg_m2 for dynamics in self.dynamics])
        self.contexts = [
            OrbitContext(dynamics.mu_km3_s2, mass, dynamics.area_m2, dynamics.cd, dynamics.cr)
            for dynamics, mass in zip(self.dynamics, self.masses)
        ]
        if np.any(~np.isfinite(self.masses)) or np.any(self.masses <= 0):
            raise ValueError("simulator.docking requires positive spacecraft masses.")
        self.environment = base_environment
        self.max_step_s = min(
            config["max_step_s"], max_step_s,
            *(step for dynamics in self.dynamics for step in
              (dynamics.orbit_substep_s, dynamics.attitude_substep_s) if step is not None),
        )
        self.phase = "approach"
        self.capture_failed = False
        self.lock_eligible_since_s: float | None = None
        self.assembly: MatedAssembly | None = None
        self.assembly_state: np.ndarray | None = None
        self.events: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self.peak_force_n = 0.
        self.peak_torque_n_m = 0.
        initial = self._pair_from_truth([agents[oid].truth for oid in self.object_ids])
        self.interface.validate_initial_geometry(initial)
        self.flight_software = None
        if config.get("commanded"):
            from sim.execution.docking_fsw import DockingFlightSoftware

            self.flight_software = DockingFlightSoftware(agents, self.object_ids, self.masses)
        self._save_sample(self._sample(0., initial))

    def prepare_initial_inputs(self, agents):
        if self.flight_software is not None:
            initial = self._pair_from_truth([agents[oid].truth for oid in self.object_ids])
            self.flight_software.prepare(initial, self.interface.geometry(initial), self.phase, 0)
            self._save_sample(self._sample(0., initial))

    @staticmethod
    def _pair_from_truth(states: list[StateTruth]) -> np.ndarray:
        return np.stack([np.concatenate((
            state.position_eci_km, state.velocity_eci_km_s,
            state.attitude_quat_bn, state.angular_rate_body_rad_s,
        )) for state in states])

    def _gravity(self, time_s: float, pair: np.ndarray) -> np.ndarray:
        return np.stack([dynamics.orbit_propagator.acceleration_at(
            t_s=time_s, x_eci=row[:6], command_accel_eci_km_s2=np.zeros(3),
            env=self.environment, ctx=context,
        ) for dynamics, row, context in zip(self.dynamics, pair, self.contexts)])

    def _advance_pair(
        self, pair: np.ndarray, time_s: float, width_s: float, *, contact: bool,
    ) -> np.ndarray:
        def derivative(t, stage):
            acceleration = self._gravity(t, stage)
            torques = np.zeros((2, 3))
            if self.flight_software is not None:
                applied_accel, torques = self.flight_software.wrenches(stage)
                acceleration += applied_accel
            if contact:
                geometry = self.interface.geometry(stage)
                forces, torques_eci, *_ = self.interface.wrenches(stage, geometry, captured=self.phase == "captured")
                acceleration += forces / self.masses[:, None] / 1000.
                torques += np.einsum("ijk,ij->ik", geometry.body_axes, torques_eci)
            rates = np.empty_like(stage)
            rates[:, :3] = stage[:, 3:6]
            rates[:, 3:6] = acceleration
            for index in range(2):
                rates[index, 6:10], rates[index, 10:13] = rigid_body_derivatives(
                    stage[index, 6:10], stage[index, 10:13],
                    self.inertias[index], torques[index], numeric_backend="python",
                )
            return rates

        following = rk4_step_state(derivative, time_s, pair, width_s)
        following[:, 6:10] /= np.linalg.norm(following[:, 6:10], axis=1)[:, None]
        if not np.all(np.isfinite(following)):
            raise FloatingPointError("Nonfinite docking state. Reduce the docking physics step.")
        return following

    def _find_contact(
        self, pair: np.ndarray, time_s: float, width_s: float, end: np.ndarray,
    ) -> tuple[float, np.ndarray] | None:
        if self.interface.geometry(end).clearance_m > 0.:
            midpoint = self._advance_pair(pair, time_s, 0.5 * width_s, contact=False)
            if self.interface.geometry(midpoint).clearance_m > 0.:
                return None
            width_s *= 0.5
            end = midpoint
        lo, hi = 0., width_s
        while hi - lo > _EVENT_TOLERANCE_S:
            middle = 0.5 * (lo + hi)
            trial = self._advance_pair(pair, time_s, middle, contact=False)
            if self.interface.geometry(trial).clearance_m <= 0.:
                hi, end = middle, trial
            else:
                lo = middle
        return hi, end

    def _sample(self, time_s, pair):
        geometry = self.interface.geometry(pair)
        force_n = torque_n_m = normal_load_n = 0.
        site_loads_n = np.zeros(4)
        force_eci_n = np.zeros(3)
        probe_torque_eci_n_m = receiver_torque_eci_n_m = np.zeros(3)
        if self.phase in {"contact", "captured"}:
            forces, torques, site_loads_n, _, _ = self.interface.wrenches(
                pair, geometry, captured=self.phase == "captured",
            )
            force_eci_n = forces[0]
            probe_torque_eci_n_m, receiver_torque_eci_n_m = torques
            force_n = float(np.linalg.norm(force_eci_n))
            torque_n_m = float(np.max(np.linalg.norm(torques, axis=1)))
            normal_load_n = float(np.sum(site_loads_n))
            self.peak_force_n = max(self.peak_force_n, force_n)
            self.peak_torque_n_m = max(self.peak_torque_n_m, torque_n_m)
        row = {
            "time_s": time_s, "phase": self.phase, "gap_m": float(geometry.offset_m[2]),
            "lateral_m": geometry.lateral_m, "angle_rad": float(np.linalg.norm(geometry.rotation_error_rad)),
            "normal_speed_m_s": float(geometry.speed_m_s[2]),
            "speed_m_s": float(np.linalg.norm(geometry.speed_m_s)),
            "rate_rad_s": float(np.linalg.norm(geometry.omega_eci[0] - geometry.omega_eci[1])),
            "normal_load_n": normal_load_n, "interface_force_n": force_n, "body_torque_n_m": torque_n_m,
            "offset_receiver_m": geometry.offset_m.tolist(),
            "rotation_error_eci_rad": geometry.rotation_error_rad.tolist(),
            "force_on_probe_eci_n": force_eci_n.tolist(), "site_loads_n": site_loads_n.tolist(),
            "probe_torque_about_com_eci_n_m": probe_torque_eci_n_m.tolist(),
            "receiver_torque_about_com_eci_n_m": receiver_torque_eci_n_m.tolist(),
        }
        if self.flight_software is not None:
            row["capture_armed"] = self.flight_software.capture_armed
            row["hard_mate_requested"] = self.flight_software.hard_mate_requested
        return row

    def _save_sample(self, row):
        if self.samples and self.samples[-1]["time_s"] == row["time_s"]:
            self.samples[-1] = row
        else:
            self.samples.append(row)

    def _transition(self, event, phase, time_s, pair):
        self.phase = phase
        row = self._sample(time_s, pair)
        self.events.append({"event": event, **row})
        self._save_sample(row)

    def _capture_if_authorized(self, pair, time_s):
        authorized = self.flight_software is None or self.flight_software.capture_armed
        if authorized and not self.capture_failed and self.interface.can_capture(self.interface.geometry(pair)):
            self._transition("soft_capture", "captured", time_s, pair)

    def _update_contact(self, pair, time_s):
        geometry = self.interface.geometry(pair)
        active_sites = geometry.radial_clearance_m <= 0.
        if geometry.facing and np.any(active_sites & (geometry.gaps_m < -self.config["max_compression_m"])):
            raise ValueError("Docking exceeded max_compression_m. The port model is outside its travel envelope.")
        row = self._sample(time_s, pair)
        self._save_sample(row)
        if self.phase == "contact" and geometry.clearance_m > 0.:
            self._transition("rebound", "separated", time_s, pair)
        if self.phase != "captured":
            return
        latch = self.config["latch"]
        *_, force, torque = self.interface.wrenches(pair, geometry, captured=True)
        if (np.linalg.norm(geometry.offset_m) > latch["max_travel_m"]
                or np.linalg.norm(force) > latch["max_force_n"]
                or np.linalg.norm(torque) > latch["max_torque_n_m"]):
            self.capture_failed = True
            self._transition("latch_release", "contact" if geometry.clearance_m <= 0. else "separated", time_s, pair)
            return
        lock = self.config.get("hard_mate")
        authorized = self.flight_software is None or self.flight_software.hard_mate_requested
        if lock and authorized and (np.linalg.norm(geometry.offset_m) <= lock["position_m"]
                     and all(row[key] <= lock[key] for key in ("angle_rad", "speed_m_s", "rate_rad_s"))):
            if self.lock_eligible_since_s is None:
                self.lock_eligible_since_s = time_s
            if time_s - self.lock_eligible_since_s >= lock["dwell_s"] - _EVENT_TOLERANCE_S:
                self.assembly, self.assembly_state = MatedAssembly.lock(pair, self.masses, self.inertias)
                self._transition("hard_mate", "mated", time_s, self.assembly.pair_state(self.assembly_state))
        else:
            self.lock_eligible_since_s = None

    def step_objects(self, inputs: list[ObjectStepInput]) -> list[ObjectStepResult]:
        start = perf_counter()
        by_id = {item.object_id: item for item in inputs}
        if set(by_id) != set(self.object_ids):
            raise RuntimeError("Docking requires both configured spacecraft to remain active.")
        ordered = [by_id[oid] for oid in self.object_ids]
        time_s, end_s = ordered[0].t_s, ordered[0].t_next
        if (ordered[1].t_s, ordered[1].t_next) != (time_s, end_s):
            raise RuntimeError("Docking spacecraft must share the same integration interval.")
        pair = self._pair_from_truth([item.initial_truth for item in ordered])
        thrust_integral = np.zeros((2, 3))
        torque_integral = np.zeros((2, 3))
        delta_v_m_s = np.zeros(2)
        max_accel = np.zeros(2)
        if self.flight_software is None:
            pair = self._advance_physics(pair, time_s, end_s)
        else:
            fsw = self.flight_software
            start_ns, stop_ns = (int(round(value * 1e9)) for value in (time_s, end_s))
            while start_ns < stop_ns:
                start_s = start_ns / 1e9
                world = fsw.prepare(pair, self.interface.geometry(pair), self.phase, start_ns)
                boundary_ns = fsw.next_boundary(start_ns, stop_ns)
                fsw.realize(world, start_ns, boundary_ns)
                if not fsw.hard_mate_requested:
                    self.lock_eligible_since_s = None
                interval_s = (boundary_ns - start_ns) / 1e9
                acceleration, torque = fsw.wrenches(pair)
                magnitudes = np.linalg.norm(acceleration, axis=1)
                thrust_integral += interval_s * acceleration
                torque_integral += interval_s * torque
                delta_v_m_s += 1000. * interval_s * magnitudes
                max_accel = np.maximum(max_accel, magnitudes)
                pair = self._advance_physics(pair, start_s, boundary_ns / 1e9)
                start_ns = boundary_ns
            fsw.prepare(pair, self.interface.geometry(pair), self.phase, stop_ns)
        self._save_sample(self._sample(end_s, pair))
        elapsed = (perf_counter() - start) / 2
        interval_s = end_s - time_s
        results = {
            oid: ObjectStepResult(
                oid, "docking_step", elapsed,
                StateTruth(row[:3].copy(), row[3:6].copy(), row[6:10].copy(), row[10:13].copy(), mass, end_s),
                thrust_integral[index] / interval_s, torque_integral[index] / interval_s,
                delta_v_m_s=float(delta_v_m_s[index]), max_accel_km_s2=float(max_accel[index]),
                burned=bool(max_accel[index] > 0.),
            )
            for index, (oid, row, mass) in enumerate(zip(self.object_ids, pair, self.masses))
        }
        return [results[item.object_id] for item in inputs]

    def _advance_physics(self, pair, time_s, end_s):
        while time_s < end_s:
            next_s = min(end_s, time_s + self.max_step_s)
            if next_s <= time_s:
                raise ValueError("Docking step is too small to advance the simulation clock.")
            width = next_s - time_s
            if self.assembly is not None:
                self.assembly_state = self.assembly.step(
                    self.assembly_state, time_s, width, self._gravity,
                    actuation=None if self.flight_software is None else self.flight_software.wrenches,
                )
                if not np.all(np.isfinite(self.assembly_state)):
                    raise FloatingPointError("Nonfinite mated state. Reduce the docking physics step.")
                self.assembly_state[6:10] = normalize_quaternion(self.assembly_state[6:10])
                pair = self.assembly.pair_state(self.assembly_state)
            else:
                contact = self.phase in {"contact", "captured"}
                following = self._advance_pair(pair, time_s, width, contact=contact)
                hit = None if contact else self._find_contact(pair, time_s, width, following)
                if hit is not None:
                    elapsed, pair = hit
                    event_s = time_s + elapsed
                    if event_s <= time_s:
                        raise ValueError("Docking event resolution cannot advance the simulation clock.")
                    time_s = min(event_s, next_s)
                    self._transition("first_contact" if not self.events else "recontact", "contact", time_s, pair)
                    self._capture_if_authorized(pair, time_s)
                    self._update_contact(pair, time_s)
                    if self.assembly is not None:
                        pair = self.assembly.pair_state(self.assembly_state)
                    continue
                pair = following
                if contact:
                    self._update_contact(pair, next_s)
                    if self.assembly is not None:
                        pair = self.assembly.pair_state(self.assembly_state)
            time_s = next_s
        return pair

    def evidence(self):
        summary = {
            "model": "rigid_ring_and_face.v1",
            "phase": self.phase, "probe_id": self.object_ids[0], "receiver_id": self.object_ids[1],
            "peak_sampled_force_n": self.peak_force_n, "peak_sampled_torque_n_m": self.peak_torque_n_m,
            "event_count": len(self.events), "max_physics_step_s": self.max_step_s,
            **({"commanded": True} if self.flight_software is not None else {}),
        }
        if self.assembly is not None:
            summary["mated_mass_kg"] = float(np.sum(self.masses))
            summary["mated_inertia_receiver_body_kg_m2"] = self.assembly.inertia_kg_m2.tolist()
        return {
            "summary": summary, "events": list(self.events), "samples": list(self.samples),
        }
