"""Bounded native passive histories consumed at ordinary scenario boundaries.

Only checked-in passive built-in forces, CR3BP and OGP providers are eligible.
The scenario still visits every sample, callback, knowledge update, impact
check, and output boundary. Commands or state changes invalidate future rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np

from sim.core.models import StateTruth
from sim.dynamics.model import OrbitalAttitudeDynamics
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.cr3bp import cr3bp_system
from sim.dynamics.orbit.epoch import TIME_DEPENDENT_ENV_CACHE_KEY
from sim.dynamics.orbit.integrators import AdaptiveStepInfo, combine_adaptive_step_info
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin, j3_plugin, j4_plugin
from sim.dynamics.orbit.rust_force_plan import immutable_builtin_plan_signature


def _optional_adaptive_float(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


@dataclass
class _PreparedHistory:
    definition: tuple[Any, ...]
    intervals: list[tuple[float, float]]
    truth_times: list[float]
    states: np.ndarray
    index: int = 0
    adaptive_infos: np.ndarray | None = None
    adaptive_method: str | None = None
    step_times: list[float] | None = None
    step_widths: list[float] | None = None
    sample_ends: list[int] | None = None


@dataclass
class _SampledSchedule:
    widths: list[float]
    times: list[float]
    ends: list[int]
    intervals: list[tuple[float, float]]
    truth_times: list[float]


class PassiveHistoryCache:
    """Reuse numerical rows without advancing the scenario's event lifecycle."""

    max_samples = 256
    max_steps = 4096

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.entries: dict[str, _PreparedHistory] = {}
        self.preparation_count = 0
        self.consumed_samples = 0




    def _environment(self, agent: Any) -> dict[str, Any]:
        dynamics = agent.dynamics
        env = {
            **self.engine.base_environment, "object_id": agent.object_id,
            "attitude_disabled": not self.engine.attitude_enabled,
            TIME_DEPENDENT_ENV_CACHE_KEY: self.engine._time_dependent_env_cache,
        }
        for key in ("drag_area_m2", "lift_area_m2"):
            value = getattr(dynamics, key)
            if value is not None and key not in env:
                env[key] = float(value)
        if dynamics.srp_area_m2 is not None:
            env["srp_area_m2"] = float(dynamics.srp_area_m2)
        return env

    def _definition(self, agent: Any, initial: StateTruth) -> tuple[Any, ...] | None:
        provider = getattr(self.engine, "general_propagation", {}).get(agent.object_id)
        if provider is not None:
            # Keep catalog-style OGP dependencies lazy.  The passive-history
            # cache also serves ordinary ONP runs, and importing the SGP4
            # provider here would eagerly load numba/scipy for every run.
            from sim.dynamics.orbit.sgp4 import SGP4EphemerisProvider
            if type(provider) is not SGP4EphemerisProvider or provider.numeric_backend != "rust":
                return None
            if not callable(getattr(provider, "canonical_states_at", None)):
                return None
            if getattr(self.engine, "collision_stepper", None) is not None or (
                getattr(self.engine, "system_force_stepper", None) is not None
            ):
                return None
            return (
                id(provider), id(provider._rust_context), float(provider.start_jd_utc),
                provider.elements, float(self.engine.dt), float(self.engine.sim_substep_s),
                None, float(self.engine.cfg.simulator.duration_s), None, "ogp", "",
            )
        dynamics = agent.dynamics
        if type(dynamics) is not OrbitalAttitudeDynamics:
            return None
        propagator = dynamics.orbit_propagator
        if type(propagator) is not OrbitPropagator or propagator.numeric_backend != "rust":
            return None
        model = str(propagator.model or "two_body").strip().lower()
        if propagator.integrator not in {"rk4", "rkf78", "adaptive", "dopri5"} or model not in {"two_body", "cr3bp"}:
            return None
        if propagator.integrator != "rk4" and model != "two_body":
            return None
        if (model == "cr3bp" and propagator.plugins) or (model == "two_body" and propagator.state_frame != "eci"):
            return None
        if dynamics.resource_model is not None or (
            dynamics.propagate_attitude and self.engine.attitude_enabled
        ):
            return None
        if dynamics.geometry_area_profile is not None or dynamics.use_rectangular_prism_for_aero_srp:
            # Geometry preparation may load frame/ephemeris resources or call
            # user-supplied area models even when this force list is passive.
            return None
        if dynamics.lift_axis_body is not None and float(dynamics.lift_coefficient) != 0.0:
            return None
        if agent.bridge is not None or agent.flight_software_runtime is not None:
            return None
        if getattr(self.engine, "collision_stepper", None) is not None or (
            getattr(self.engine, "system_force_stepper", None) is not None
        ):
            return None
        plan_signature = None
        simple_passive = not propagator.plugins or (
            len(propagator.plugins) == 1 and propagator.plugins[0] is j2_plugin
        )
        if model == "two_body" and not simple_passive:
            if all(any(plugin is known for known in (j2_plugin, j3_plugin, j4_plugin))
                   for plugin in propagator.plugins):
                # These exact built-ins read only mu and state. Bind their
                # order without repeatedly hashing unrelated frame/weather
                # resources at every ordinary lifecycle boundary.
                plan_signature = ("zonal", tuple(propagator.plugins))
            else:
                ctx = OrbitContext(dynamics.mu_km3_s2, initial.mass_kg, dynamics.area_m2, dynamics.cd, dynamics.cr)
                plan_signature = immutable_builtin_plan_signature(propagator, self._environment(agent), ctx)
                if plan_signature is None:
                    return None
            model = "force_plan"
        if propagator.integrator in {"rkf78", "adaptive", "dopri5"}:
            adaptive_kind = "adaptive_force_plan" if model == "force_plan" and not all(
                any(plugin is known for known in (j2_plugin, j3_plugin, j4_plugin))
                for plugin in propagator.plugins
            ) else "adaptive_zonal"
            plan_signature = (
                adaptive_kind, propagator.integrator, tuple(propagator.plugins), plan_signature,
                float(propagator.adaptive_atol), float(propagator.adaptive_rtol),
            )
            model = adaptive_kind
        return (
            id(dynamics), id(propagator), float(dynamics.mu_km3_s2),
            bool(propagator.plugins), float(self.engine.dt),
            float(self.engine.sim_substep_s), dynamics.orbit_substep_s,
            float(self.engine.cfg.simulator.duration_s), plan_signature,
            model, str(propagator.cr3bp_system_name),
        )

    def _prepare(
        self, definition: tuple[Any, ...], initial: StateTruth, t_s: float, t_next: float, agent: Any,
    ) -> _PreparedHistory | None:
        _, _, mu, j2, dt, sim_substep, orbit_substep, duration, _, model, system_name = definition
        if model == "ogp":
            return self._prepare_ogp(definition, initial, t_s, t_next)
        native = import_module("oel_rust_orbit")
        propagate = getattr(native, (
            "cr3bp_sampled_history_bytes" if model == "cr3bp"
            else "propagate_passive_sampled_eci_bytes"
        ), None)
        if propagate is None and model not in {"force_plan", "adaptive_zonal", "adaptive_force_plan"}:
            # Compatible older wheels retain the ordinary Rust stepper.
            return None
        schedule = self._schedule(definition, initial, t_s, t_next)
        if schedule is None:
            return None
        widths, times, ends = schedule.widths, schedule.times, schedule.ends
        intervals, truth_times = schedule.intervals, schedule.truth_times
        initial_vector = np.concatenate((initial.position_eci_km, initial.velocity_eci_km_s))
        try:
            packed_widths = np.asarray(widths, dtype="<f8").tobytes()
            adaptive_infos = None
            if model == "adaptive_zonal":
                propagator = agent.dynamics.orbit_propagator
                codes = [{j2_plugin: 2, j3_plugin: 3, j4_plugin: 4}[plugin]
                         for plugin in propagator.plugins]
                adaptive = getattr(native, "propagate_adaptive_sampled_eci_bytes", None)
                if adaptive is None:
                    return None
                h_init = propagator._rkf78_h_next
                if (propagator._rkf78_last_t_s is None
                        or times[0] < float(propagator._rkf78_last_t_s) - 1.0e-12):
                    h_init = None
                raw, info_raw = adaptive(
                    initial_vector.tolist(), np.asarray(times, dtype="<f8").tobytes(),
                    packed_widths, ends, mu, codes,
                    float(propagator.adaptive_atol), float(propagator.adaptive_rtol),
                    h_init,
                    1 if propagator.integrator == "dopri5" else 0,
                )
                adaptive_infos = np.frombuffer(info_raw, dtype="<f8").copy().reshape(len(widths), 8)
            elif model == "adaptive_force_plan":
                dynamics = agent.dynamics
                ctx = OrbitContext(mu, initial.mass_kg, dynamics.area_m2, dynamics.cd, dynamics.cr)
                prepared = dynamics.orbit_propagator.try_propagate_adaptive_sampled_steps(
                    initial_vector, widths, ends, initial.t_s, np.zeros(3),
                    self._environment(agent), ctx, step_times=times,
                )
                if prepared is None:
                    return None
                states, adaptive_infos = prepared
            elif model == "force_plan":
                dynamics = agent.dynamics
                ctx = OrbitContext(mu, initial.mass_kg, dynamics.area_m2, dynamics.cd, dynamics.cr)
                states = dynamics.orbit_propagator.try_propagate_sampled_steps(
                    initial_vector, widths, ends, initial.t_s, np.zeros(3),
                    self._environment(agent), ctx, step_times=times,
                )
                if states is None:
                    return None
            elif model == "cr3bp":
                system = cr3bp_system(system_name)
                raw = propagate(
                    initial_vector.tolist(), packed_widths, ends,
                    system.distance_km, system.mu, system.mean_motion_rad_s,
                )
            else:
                raw = propagate(initial_vector.tolist(), packed_widths, ends, mu, j2)
        except (ValueError, ArithmeticError, RuntimeError):
            # A future numerical failure must not surface before an earlier
            # termination boundary. The ordinary current step remains owner.
            return None
        if model not in {"force_plan", "adaptive_force_plan"}:
            states = np.frombuffer(raw, dtype="<f8").copy().reshape(len(ends) + 1, 6)
        self.preparation_count += 1
        return _PreparedHistory(
            definition, intervals, truth_times, states, adaptive_infos=adaptive_infos,
            adaptive_method=("dopri5" if agent.dynamics.orbit_propagator.integrator == "dopri5" else "rkf78")
            if adaptive_infos is not None else None,
            step_times=times if adaptive_infos is not None else None,
            step_widths=widths if adaptive_infos is not None else None,
            sample_ends=ends if adaptive_infos is not None else None,
        )


    def _schedule(
        self, definition: tuple[Any, ...], initial: StateTruth, t_s: float, t_next: float,
    ) -> _SampledSchedule | None:
        sim_substep, orbit_substep, duration = definition[5:8]
        # Game speed controls can override the configured outer step. Prepare
        # future rows at the observed cadence; existing guards handle changes.
        dt = float(t_next) - float(t_s)
        widths: list[float] = []
        times: list[float] = []
        ends: list[int] = []
        intervals: list[tuple[float, float]] = []
        truth_times = [float(initial.t_s)]
        truth_clock = float(initial.t_s)
        start, stop = float(t_s), float(t_next)
        for _ in range(self.max_samples):
            interval_widths: list[float] = []
            interval_times: list[float] = []
            current = start
            next_truth_clock = truth_clock
            while current < stop:
                h = min(max(sim_substep, 1.0e-12), stop - current)
                inner_h = OrbitalAttitudeDynamics._effective_substep(orbit_substep, h)
                if len(widths) + len(interval_widths) + int(h / inner_h) + 1 > self.max_steps:
                    break
                local_clock = next_truth_clock
                for inner_width in OrbitalAttitudeDynamics._substep_sequence(h, inner_h):
                    interval_times.append(local_clock)
                    interval_widths.append(inner_width)
                    local_clock += float(inner_width)
                current += h
                next_truth_clock += h
            if current < stop:
                break
            widths.extend(interval_widths)
            times.extend(interval_times)
            ends.append(len(widths))
            intervals.append((start, stop))
            truth_clock = next_truth_clock
            truth_times.append(truth_clock)
            start = stop
            stop = float(start + min(dt, max(duration - start, 0.0)))
            if stop <= start:
                break
        if not intervals:
            return None
        return _SampledSchedule(widths, times, ends, intervals, truth_times)

    def _prepare_ogp(
        self, definition: tuple[Any, ...], initial: StateTruth, t_s: float, t_next: float,
    ) -> _PreparedHistory | None:
        provider = next(
            item for item in self.engine.general_propagation.values() if id(item) == definition[0]
        )
        duration = definition[7]
        dt = float(t_next) - float(t_s)
        intervals = []
        truth_times = [float(initial.t_s)]
        start, stop = float(t_s), float(t_next)
        for _ in range(self.max_samples):
            intervals.append((start, stop))
            truth_times.append(stop)
            start = stop
            stop = float(start + min(dt, max(duration - start, 0.0)))
            if stop <= start:
                break
        try:
            states = provider.canonical_states_at(truth_times)
        except (ValueError, ArithmeticError, RuntimeError):
            # Preserve current-step failure/termination order when an error
            # belongs to a later speculative epoch.
            return None
        self.preparation_count += 1
        return _PreparedHistory(definition, intervals, truth_times, states)

    def _matching_entry(
        self, aid: str, definition: tuple[Any, ...], initial: StateTruth, t_s: float, t_next: float,
    ) -> _PreparedHistory | None:
        state = np.concatenate((initial.position_eci_km, initial.velocity_eci_km_s))
        prepared = self.entries.get(aid)
        if prepared is not None and (
            prepared.definition != definition
            or prepared.index >= len(prepared.intervals)
            or prepared.intervals[prepared.index] != (float(t_s), float(t_next))
            or prepared.truth_times[prepared.index] != float(initial.t_s)
            or not np.array_equal(prepared.states[prepared.index], state)
        ):
            prepared = None
            self.entries.pop(aid, None)
        return prepared

    def step(
        self, *, aid: str, agent: Any, initial: StateTruth, t_s: float, t_next: float,
    ) -> StateTruth | None:
        definition = self._definition(agent, initial)
        if definition is None:
            self.entries.pop(aid, None)
            return None
        prepared = self._matching_entry(aid, definition, initial, t_s, t_next)
        if prepared is None:
            started = perf_counter()
            prepared = self._prepare(definition, initial, t_s, t_next, agent)
            self.engine.runtime_profiler.record_stage("passive_history_prepare", perf_counter() - started)
            if prepared is None:
                return None
            self.entries[aid] = prepared
        prepared.index += 1
        self.consumed_samples += 1
        vector = prepared.states[prepared.index]
        if definition[-2] == "ogp":
            provider = self.engine.general_propagation[aid]
            return StateTruth(
                position_eci_km=vector[:3].copy(), velocity_eci_km_s=vector[3:].copy(),
                attitude_quat_bn=provider._attitude_quat(), angular_rate_body_rad_s=provider._angular_rate_body(),
                mass_kg=float(provider.mass_kg), t_s=prepared.truth_times[prepared.index],
            )
        propagator = agent.dynamics.orbit_propagator
        propagator.last_numeric_path = "rust_native_passive_history"
        if prepared.adaptive_infos is not None:
            assert prepared.sample_ends is not None and prepared.step_times is not None
            assert prepared.step_widths is not None
            first = 0 if prepared.index == 1 else prepared.sample_ends[prepared.index - 2]
            last = prepared.sample_ends[prepared.index - 1]
            for step_index in range(first, last):
                row = prepared.adaptive_infos[step_index]
                assert prepared.adaptive_method is not None
                info = AdaptiveStepInfo(
                    method=prepared.adaptive_method, accepted_steps=int(row[0]), rejected_steps=int(row[1]),
                    attempted_steps=int(row[2]), min_step_s=_optional_adaptive_float(row[3]),
                    max_step_s=_optional_adaptive_float(row[4]), final_step_s=_optional_adaptive_float(row[5]),
                    suggested_next_step_s=_optional_adaptive_float(row[6]), max_error_ratio=float(row[7]),
                )
                propagator._rkf78_h_next = info.suggested_next_step_s
                propagator._rkf78_last_t_s = (
                    prepared.step_times[step_index] + prepared.step_widths[step_index]
                )
                propagator.last_adaptive_step_info = info
                previous = [] if propagator.adaptive_step_info is None else [propagator.adaptive_step_info]
                propagator.adaptive_step_info = combine_adaptive_step_info(
                    prepared.adaptive_method, [*previous, info],
                )
        if definition[-2] == "cr3bp":
            propagator = agent.dynamics.orbit_propagator
            propagator.last_adaptive_step_info = None
            if propagator._rkf78_last_t_s is None or float(initial.t_s) < float(propagator._rkf78_last_t_s) - 1e-12:
                propagator._rkf78_h_next = None
        return StateTruth(
            position_eci_km=vector[:3].copy(), velocity_eci_km_s=vector[3:].copy(),
            attitude_quat_bn=initial.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=initial.angular_rate_body_rad_s.copy(),
            mass_kg=max(0.0, float(initial.mass_kg)),
            t_s=prepared.truth_times[prepared.index],
        )
