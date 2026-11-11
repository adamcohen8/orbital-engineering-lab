"""Passive Earth ECI ONP trials for bounded impulsive trajectory observations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from sim.config import SimulationScenarioConfig
from sim.single_run import _SingleRunEngine


def prepare_onp_scenario(
    template: dict[str, Any], initial_state_eci: np.ndarray, output_dir: Path,
    *, coast_step_s: float, duration_s: float,
) -> dict[str, Any]:
    """Keep a template's force configuration and install one passive trial state."""
    source = copy.deepcopy(template)
    objects = source.get("objects", {})
    if set(objects) != {"vehicle"}:
        raise ValueError("ONP targeting requires exactly one object named vehicle")
    vehicle = objects["vehicle"]
    if (vehicle.get("kind") != "satellite" or not vehicle.get("enabled", True)
            or vehicle.get("runtime_profile") != "trajectory_only"):
        raise ValueError("ONP targeting requires one enabled trajectory_only satellite")
    simulator = source.get("simulator", {})
    orbit = simulator.get("dynamics", {}).get("orbit", {})
    if orbit.get("model", "two_body") != "two_body":
        raise ValueError("ONP targeting requires Earth ECI ONP with the two_body base and optional force plugins")
    if simulator.get("dynamics", {}).get("attitude", {}).get("enabled", True):
        raise ValueError("ONP targeting requires attitude dynamics disabled")
    if (vehicle.get("general") or vehicle.get("deploy") or vehicle.get("flight_software")
            or vehicle.get("propagation_method", "") not in {"", "special"}):
        raise ValueError("ONP targeting requires a passive numerical satellite")
    state = np.asarray(initial_state_eci, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("ONP targeting requires a finite six-component ECI state")
    vehicle["initial_state"] = {
        "position_eci_km": state[:3].tolist(),
        "velocity_eci_km_s": state[3:].tolist(),
    }
    simulator["duration_s"] = float(duration_s)
    simulator["dt_s"] = float(coast_step_s)
    simulator.setdefault("termination", {})["earth_impact_enabled"] = False
    source["scenario_name"] = "trajectory_target_impulsive_onp"
    source["scenario_description"] = "Exact impulses with configured OEL ONP forces."
    outputs = source.setdefault("outputs", {})
    outputs["output_dir"] = str(output_dir)
    outputs["mode"] = "save"
    outputs["review"] = {"enabled": True, "detail": "standard"}
    outputs["plots"] = {"enabled": False, "figure_ids": []}
    outputs["animations"] = {"enabled": False, "types": []}
    return source


class ONPTrial:
    """One fresh OEL trajectory-only runtime, including its time-dependent environment."""

    def __init__(self, cfg: SimulationScenarioConfig) -> None:
        self.engine = _SingleRunEngine(cfg)
        self.agent = self.engine.agents["vehicle"]
        self.truth = self.agent.truth.copy()

    @property
    def time_s(self) -> float:
        return float(self.truth.t_s)

    def state(self) -> np.ndarray:
        return np.concatenate((self.truth.position_eci_km, self.truth.velocity_eci_km_s))

    def _propagate(self, truth: Any, seconds: float) -> Any:
        if seconds < 0.0:
            raise ValueError("ONP coast duration cannot be negative")
        if seconds == 0.0:
            return truth.copy()
        return self.engine.satellite_stepper._step_trajectory_only(
            aid="vehicle", agent=self.agent, initial_truth=truth,
            world_truth_decision={"vehicle": truth}, t_s=truth.t_s,
            t_next=truth.t_s + seconds,
        ).truth

    def coast(self, seconds: float) -> np.ndarray:
        self.truth = self._propagate(self.truth, seconds)
        return self.state()

    def impulse(self, delta_v_eci_m_s: np.ndarray) -> np.ndarray:
        dv = np.asarray(delta_v_eci_m_s, dtype=float)
        if dv.shape != (3,) or not np.isfinite(dv).all() or np.linalg.norm(dv) > 100_000.0:
            raise ValueError("ONP impulse must be a finite ECI three-vector within 100 km/s")
        self.truth.velocity_eci_km_s += dv / 1000.0
        return self.state()

    def first_apsis(self, kind: str, *, step_s: float, max_time_s: float) -> tuple[float, np.ndarray]:
        if kind not in {"apogee", "perigee"}:
            raise ValueError("kind must be apogee or perigee")
        sign = 1.0 if kind == "apogee" else -1.0
        left = self.truth.copy()
        radial_left = sign * float(np.dot(left.position_eci_km, left.velocity_eci_km_s))
        while left.t_s < max_time_s:
            dt = min(step_s, max_time_s - left.t_s)
            if dt <= 0.0:
                break
            right = self._propagate(left, dt)
            radial_right = sign * float(np.dot(right.position_eci_km, right.velocity_eci_km_s))
            if radial_left > 0.0 >= radial_right:
                lo, hi = 0.0, dt
                for _ in range(36):
                    mid = (lo + hi) / 2.0
                    at_mid = self._propagate(left, mid)
                    if sign * float(np.dot(at_mid.position_eci_km, at_mid.velocity_eci_km_s)) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                self.truth = self._propagate(left, (lo + hi) / 2.0)
                return self.time_s, self.state()
            left, radial_left = right, radial_right
        raise ValueError(f"No first {kind} event occurred within max_time_s")

    def close(self) -> None:
        self.engine.object_step_executor.shutdown()
