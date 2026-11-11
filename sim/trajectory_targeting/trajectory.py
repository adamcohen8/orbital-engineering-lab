"""Bounded OEL trajectory observation adapter for vector differential correction."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sim.config import scenario_config_from_dict, validate_scenario_plugins
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.trajectory_targeting.apogee import _first_apsis
from sim.trajectory_targeting.onp_adapter import ONPTrial, prepare_onp_scenario

EventKind = Literal["fixed_time", "first_apogee", "first_perigee"]
Frame = Literal["eci", "ric"]
Metric = Literal[
    "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s",
    "radius_km", "speed_km_s", "radial_speed_km_s",
]


@dataclass(frozen=True)
class Impulse:
    maneuver_id: str
    event: EventKind
    time_s: float | None
    frame: Frame
    components_m_s: tuple[float, float, float]


@dataclass(frozen=True)
class Observation:
    event: EventKind
    time_s: float | None
    metrics: tuple[Metric, ...]


@dataclass(frozen=True)
class TrajectoryPlan:
    initial_state_eci: tuple[float, float, float, float, float, float]
    impulses: tuple[Impulse, ...]
    observation: Observation
    coast_step_s: float = 30.0
    max_time_s: float = 20_000.0


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    state_eci: tuple[float, ...]
    metrics: tuple[float, ...]
    maneuvers: tuple[dict, ...]


def observe_impulsive_trajectory(
    plan: TrajectoryPlan,
    *,
    vary: dict[tuple[str, str], float] | None = None,
    onp_template: dict[str, Any] | None = None,
) -> TrajectorySample:
    """Apply up to four exact impulses and observe one OEL-propagated event.

    `vary` keys are `(maneuver_id, "time_s" | "component_0_m_s" |
    "component_1_m_s" | "component_2_m_s")`. The component axes are ECI or
    RIC according to that impulse's frame. Pass the returned metrics to
    `solve_vector_target` with explicit variable bounds and goal tolerances.
    A trusted `onp_template` selects OEL's configured Earth ECI ONP forces;
    without one, the historical point-mass RK4 path is retained.
    """
    state = np.asarray(plan.initial_state_eci, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all() or np.linalg.norm(state[:3]) <= EARTH_RADIUS_KM:
        raise ValueError("Initial state must be a finite six-element ECI state above Earth")
    if not (0.1 <= plan.coast_step_s <= 300.0 and plan.coast_step_s < plan.max_time_s <= 86_400.0
            and plan.max_time_s / plan.coast_step_s <= 5000):
        raise ValueError("Trajectory coast limits are invalid")
    if len(plan.impulses) > 4 or len({burn.maneuver_id for burn in plan.impulses}) != len(plan.impulses):
        raise ValueError("Trajectory requires at most four uniquely named impulses")
    if not plan.observation.metrics:
        raise ValueError("Trajectory observation requires at least one metric")
    changes = dict(vary or {})
    known = {(burn.maneuver_id, field) for burn in plan.impulses
             for field in ("time_s", "component_0_m_s", "component_1_m_s", "component_2_m_s")}
    if set(changes) - known:
        raise ValueError("Unknown trajectory vary key")
    if not all(math.isfinite(float(value)) for value in changes.values()):
        raise ValueError("Trajectory vary values must be finite")
    if onp_template is None:
        return _observe_plan(plan, state, changes, None)
    with tempfile.TemporaryDirectory(prefix="oel-onp-observation-") as scratch:
        duration = math.ceil(plan.max_time_s / plan.coast_step_s) * plan.coast_step_s + plan.coast_step_s
        scenario = prepare_onp_scenario(
            onp_template, state, Path(scratch), coast_step_s=plan.coast_step_s, duration_s=duration)
        cfg = scenario_config_from_dict(scenario)
        plugin_errors = validate_scenario_plugins(cfg)
        if plugin_errors:
            raise ValueError("ONP template plugin validation failed: " + "; ".join(plugin_errors))
        trial = ONPTrial(cfg)
        try:
            return _observe_plan(plan, state, changes, trial)
        finally:
            trial.close()


def _observe_plan(
    plan: TrajectoryPlan, state: np.ndarray, changes: dict[tuple[str, str], float], trial: ONPTrial | None,
) -> TrajectorySample:

    def coast(current: np.ndarray, duration_s: float) -> np.ndarray:
        if duration_s < -1e-9:
            raise ValueError("Maneuver or observation time precedes the current event")
        n = max(1, math.ceil(max(duration_s, 0.0) / plan.coast_step_s))
        if duration_s <= 0:
            return current
        dt = duration_s / n
        for _ in range(n):
            current = (propagate_two_body_rk4(current, dt, EARTH_MU_KM3_S2, np.zeros(3))
                       if trial is None else trial.coast(dt))
        return current

    def advance(event: EventKind, fixed_time: float | None, current: np.ndarray, t_s: float) -> tuple[np.ndarray, float]:
        if event == "fixed_time":
            if fixed_time is None or not math.isfinite(fixed_time) or not t_s <= fixed_time <= plan.max_time_s:
                raise ValueError("Fixed event time is missing, past, or beyond max_time_s")
            return coast(current, fixed_time - t_s), fixed_time
        if event not in {"first_apogee", "first_perigee"}:
            raise ValueError("Unsupported trajectory event")
        if fixed_time is not None:
            raise ValueError("Apsis event cannot also have a fixed time")
        kind = "apogee" if event == "first_apogee" else "perigee"
        if trial is not None:
            event_time, observed = trial.first_apsis(kind, step_s=plan.coast_step_s, max_time_s=plan.max_time_s)
            return observed, event_time
        event_dt, observed = _first_apsis(
            current, kind=kind, step_s=plan.coast_step_s, max_coast_s=plan.max_time_s - t_s)
        return observed, t_s + event_dt

    time_s = 0.0
    maneuver_rows: list[dict] = []
    for burn in plan.impulses:
        if not burn.maneuver_id or burn.frame not in {"eci", "ric"}:
            raise ValueError("Impulse requires an id and ECI or RIC frame")
        burn_time = changes.get((burn.maneuver_id, "time_s"), burn.time_s)
        if burn.event != "fixed_time" and (burn.maneuver_id, "time_s") in changes:
            raise ValueError("Cannot vary time of an orbital-event impulse")
        state, time_s = advance(burn.event, burn_time, state, time_s)
        components = np.asarray([
            changes.get((burn.maneuver_id, f"component_{index}_m_s"), original)
            for index, original in enumerate(burn.components_m_s)
        ], dtype=float)
        if components.shape != (3,) or not np.isfinite(components).all() or np.linalg.norm(components) > 100_000:
            raise ValueError("Impulse components must be finite and within 100 km/s")
        if burn.frame == "ric":
            radial = state[:3] / np.linalg.norm(state[:3])
            angular_momentum = np.cross(state[:3], state[3:])
            h_norm = float(np.linalg.norm(angular_momentum))
            if h_norm <= 1e-10:
                raise ValueError("RIC frame is undefined for near-radial motion")
            cross_track = angular_momentum / h_norm
            in_track = np.cross(cross_track, radial)
            dv_eci = radial * components[0] + in_track * components[1] + cross_track * components[2]
        else:
            dv_eci = components
        pre = state.copy()
        state = state.copy()
        state[3:] += dv_eci / 1000.0
        if trial is not None:
            trial.impulse(dv_eci)
        maneuver_rows.append({
            "maneuver_id": burn.maneuver_id, "time_s": time_s, "frame": burn.frame,
            "components_m_s": components.tolist(), "delta_v_eci_m_s": dv_eci.tolist(),
            "pre_state_eci": pre.tolist(), "post_state_eci": state.tolist(),
        })
    state, time_s = advance(plan.observation.event, plan.observation.time_s, state, time_s)
    if not np.isfinite(state).all():
        raise ValueError("Trajectory observation is nonfinite")
    radius = float(np.linalg.norm(state[:3]))
    measures = {
        "x_km": state[0], "y_km": state[1], "z_km": state[2],
        "vx_km_s": state[3], "vy_km_s": state[4], "vz_km_s": state[5],
        "radius_km": radius, "speed_km_s": np.linalg.norm(state[3:]),
        "radial_speed_km_s": np.dot(state[:3], state[3:]) / radius,
    }
    try:
        metrics = tuple(float(measures[name]) for name in plan.observation.metrics)
    except KeyError as exc:
        raise ValueError(f"Unsupported trajectory observation metric: {exc.args[0]}") from exc
    return TrajectorySample(time_s, tuple(float(value) for value in state), metrics, tuple(maneuver_rows))
