"""Integrate a rocket through first contact and settling with held commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sim.rocket.landing_gear import LandingGearCommand
from sim.rocket.touchdown import classify_touchdown

if TYPE_CHECKING:
    from sim.rocket.engine import RocketAscentSimulator
    from sim.rocket.landing_gear import ContactPoint
    from sim.rocket.models import GuidanceCommand, RocketState


def _strike_reason(point: ContactPoint) -> str | None:
    if point.leg is not None and point.status == "down_locked":
        return "leg_stroke_exceeded" if point.gap_m < -point.leg.max_stroke_m else None
    if point.gap_m <= 1e-6:
        return "body_strike" if point.leg is None else "gear_not_locked"
    return None


def _closing_contacts(points: list[ContactPoint]) -> list[ContactPoint]:
    return [
        point for point in points if point.gap_m < -1e-6 or (
            point.gap_m <= 1e-6 and point.velocity_ecef_m_s @ point.normal_ecef < -1e-8
        )
    ]


def advance_landing(
    simulator: RocketAscentSimulator, initial: RocketState, command: GuidanceCommand, throttle: float, dt_s: float,
) -> RocketState:
    model = simulator.gear_model
    if model is None:
        raise RuntimeError("Landing integration requires a gear model.")
    state = initial.copy()
    if not state.landing_legs:
        state.landing_legs = model.initial_legs()
    if state.landing_legs.keys() != model.legs_by_id.keys():
        raise ValueError("RocketState landing legs must match the configured hardware.")
    if state.gear_terminal is not None:
        return state
    schedule = model.config.commands
    command_time = state.t_s
    end = state.t_s + dt_s
    tolerance_s = 1e-10
    thrust_impulse = np.zeros(3)
    thrust_n_s = 0.0
    torque_impulse = np.zeros(3)
    elapsed = 0.0

    while state.t_s < end:
        if not np.isfinite(state.mass_kg) or state.mass_kg <= 0.0:
            raise ValueError("Landing gear contact requires positive rocket mass.")
        due = tuple(LandingGearCommand(item.leg_id, item.action) for item in schedule
                    if abs(item.time_s - state.t_s) <= tolerance_s)
        if state.t_s == command_time:
            # Live commands win for the same actuator at the same boundary.
            live = command.landing_gear_commands
            live_ids = {item.leg_id for item in live if isinstance(item, LandingGearCommand)}
            due = tuple(item for item in due if item.leg_id not in live_ids) + live
        model.command(state, due)
        points, body_to_ecef, eci_to_ecef = model.sample(state)
        if state.gear_touchdown is None and any(point.gap_m < -1e-5 for point in points):
            raise ValueError("Rocket landing gear starts below the configured contact surface.")
        next_command = min((item.time_s for item in schedule if item.time_s > state.t_s + tolerance_s), default=end)
        remaining = end - state.t_s
        h = min(remaining, next_command - state.t_s, model.motion_boundary(state),
                model.contact_step(state.mass_kg, points))
        if state.gear_settle_elapsed_s > 0.0:
            h = min(h, model.config.settle_time_s - state.gear_settle_elapsed_s)
        if h <= tolerance_s and remaining > tolerance_s:
            raise RuntimeError("Landing gear contact step became too small. Check inertia and contact coefficients.")
        if state.t_s + h == state.t_s:
            raise ValueError("Landing gear step cannot advance the rocket clock.")
        wrench = model.wrench(points, body_to_ecef, eci_to_ecef) if state.gear_touchdown else None

        def propagate(interval, start=state, ground_wrench=wrench):
            trial = simulator._step_once(start, command, throttle, interval, ground_wrench=ground_wrench)
            # Leg motion is evaluated over the same trial interval as the rigid body.
            model.advance(trial, interval)
            return trial

        trial = propagate(h)
        trial_points = model.sample(trial)[0]
        touched = state.gear_touchdown is not None

        def critical(sample, after_touchdown=touched):
            return [point for point in sample if _strike_reason(point)] if after_touchdown else _closing_contacts(sample)

        hits = critical(trial_points)
        if hits:
            lo, hi = 0.0, h
            while hi - lo > 1e-7:
                mid = (lo + hi) / 2.0
                midpoint = propagate(mid)
                midpoint_points = model.sample(midpoint)[0]
                if critical(midpoint_points):
                    hi, trial, trial_points = mid, midpoint, midpoint_points
                else:
                    lo = mid
            h = hi
            hits = critical(trial_points)
            # Give a body strike precedence over a coincident footpad contact.
            point = min(hits, key=lambda item: (item.leg is not None, item.status == "down_locked", item.gap_m, item.id))
            failure = _strike_reason(point)
            if not touched:
                event = classify_touchdown(
                    trial, limits=simulator.sim_cfg.touchdown, context=model.context, use_ellipsoid=model.ellipsoid,
                    upright_axis_body=simulator.vehicle_cfg.thrust_axis_body, bracket_start_s=state.t_s + lo,
                    point=point, failure=failure,
                )
                trial.gear_touchdown = {
                    **event.evidence,
                    "outcome": "contact" if event.outcome == "landing" else "crash",
                    "contact_point_id": point.id,
                    "contact_point_status": point.status,
                    "contact_position_ecef_km": point.position_ecef_km.tolist(),
                    "contact_velocity_ecef_m_s": point.velocity_ecef_m_s.tolist(),
                    "contact_normal_closing_speed_m_s": max(0.0, -float(point.velocity_ecef_m_s @ point.normal_ecef)),
                    "gear_status": {key: leg.status for key, leg in trial.landing_legs.items()},
                }
                if event.outcome == "crash":
                    trial.gear_terminal = {
                        "outcome": "crash", "reason": failure or "touchdown_limits_exceeded",
                        "time_s": trial.t_s, "contact_point_id": point.id,
                    }
            elif failure is not None:
                trial.gear_terminal = {
                    "outcome": "crash", "reason": failure,
                    "time_s": trial.t_s, "contact_point_id": point.id,
                }

        if trial.gear_touchdown is not None:
            model.update_loads(trial, trial_points)
            if (touched and trial.gear_rebound_time_s is None
                    and any(point.leg is not None and point.gap_m <= 0.0 for point in points)
                    and all(point.gap_m > 0.0 for point in trial_points)):
                trial.gear_rebound_time_s = trial.t_s
            if trial.gear_terminal is None:
                continuously_supported = touched and model.supported(state, points) and model.supported(trial, trial_points)
                trial.gear_settle_elapsed_s = state.gear_settle_elapsed_s + h if continuously_supported else 0.0
                if trial.gear_settle_elapsed_s >= model.config.settle_time_s - tolerance_s:
                    trial.gear_terminal = {"outcome": "landing", "reason": "settled", "time_s": trial.t_s}

        thrust_impulse += trial._last_step_thrust_accel_eci_km_s2 * h
        torque_impulse += trial._last_step_torque_body_nm * h
        thrust_n_s += trial._last_step_thrust_n * h
        elapsed += h
        state = trial
        if state.gear_terminal is not None:
            break

    if elapsed > 0.0:
        state._last_step_thrust_accel_eci_km_s2 = thrust_impulse / elapsed
        state._last_step_torque_body_nm = torque_impulse / elapsed
        state._last_step_thrust_n = thrust_n_s / elapsed
    return state


def advance_held_gear(simulator: RocketAscentSimulator, state: RocketState, t_s: float) -> None:
    """Advance scheduled gear while the vehicle is held on the launch pad."""
    model = simulator.gear_model
    if model is None:
        return
    for entry in model.config.commands:
        if state.t_s - 1e-10 <= entry.time_s < t_s - 1e-10:
            interval = max(0.0, entry.time_s - state.t_s)
            state.t_s += interval
            model.advance(state, interval)
            model.command(state, (LandingGearCommand(entry.leg_id, entry.action),))
    interval = t_s - state.t_s
    state.t_s = t_s
    model.advance(state, interval)
