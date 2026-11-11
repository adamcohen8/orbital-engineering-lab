"""Optional Rust evaluation for ONP Cartesian observation arcs.

The adapter prepares observation epochs, references, and Cholesky factors once
per fit. Each optimizer candidate then crosses the native boundary once and
returns exact-epoch states plus the already whitened residual vector. The
Python solver remains authoritative for parameter mapping, robust loss, priors,
clipping, holdout partitioning, and artifacts. Unsupported batched force plans
stay with the configured exact-epoch session evaluator.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from functools import lru_cache
from importlib import import_module
from typing import Sequence

import numpy as np


@lru_cache(maxsize=1)
def _extension():
    try:
        return import_module("oel_rust_orbit")
    except ImportError as exc:  # pragma: no cover - exercised by installation checks
        raise RuntimeError(
            "Rust ONP OD backend requested but oel_rust_orbit is unavailable; "
            "install the optional oel_rust_orbit wheel first"
        ) from exc


def _f64_bytes(value: Sequence[float] | np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(value, dtype="<f8")).tobytes()


def _f64_from_bytes(value: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(value, dtype="<f8").copy().reshape(shape)


class RustONPObservationContext:
    """Persistent native ONP observation and whitening context."""

    backend_name = "rust_onp_observation_context"

    def __init__(
        self,
        times_s: Sequence[float] | np.ndarray,
        reference_positions_km: Sequence[Sequence[float]] | np.ndarray,
        reference_velocities_km_s: Sequence[Sequence[float]] | np.ndarray | None,
        factor_blocks: Sequence[np.ndarray],
        dimensions: Sequence[int],
        step_widths_s: Sequence[float] | np.ndarray,
        sample_end_indices: Sequence[int],
        *,
        mu_km3_s2: float,
        include_j2: bool,
    ) -> None:
        native_class = getattr(_extension(), "ONPObservationContext", None)
        if native_class is None:
            raise RuntimeError(
                "installed oel_rust_orbit wheel does not provide ONPObservationContext; "
                "rebuild the wheel with the fused ONP OD kernels"
            )
        self.times_s = np.ascontiguousarray(np.asarray(times_s, dtype=np.float64)).reshape(-1)
        self.positions = np.ascontiguousarray(np.asarray(reference_positions_km, dtype=np.float64))
        self.velocities = (
            np.ascontiguousarray(np.asarray(reference_velocities_km_s, dtype=np.float64))
            if reference_velocities_km_s is not None
            else np.empty((0, 3), dtype=np.float64)
        )
        self.dimensions = [int(value) for value in dimensions]
        if self.times_s.size == 0 or not np.all(np.isfinite(self.times_s)) or np.any(self.times_s < 0.0):
            raise ValueError("ONP observation epochs must be finite and non-negative")
        if np.any(np.diff(self.times_s) < 0.0):
            raise ValueError("ONP observation epochs must be non-decreasing")
        rows = self.times_s.size
        if self.positions.shape != (rows, 3) or not np.all(np.isfinite(self.positions)):
            raise ValueError("reference positions must have shape (rows, 3) and be finite")
        if self.velocities.size and (
            self.velocities.shape != (rows, 3) or not np.all(np.isfinite(self.velocities))
        ):
            raise ValueError("reference velocities must have shape (rows, 3) and be finite")
        if len(self.dimensions) != rows or any(value not in (3, 6) for value in self.dimensions):
            raise ValueError("ONP observation dimensions must contain one 3 or 6 entry per epoch")
        if self.velocities.size == 0 and any(value == 6 for value in self.dimensions):
            raise ValueError("six-component observations require reference velocities")
        if self.velocities.size and any(value != 6 for value in self.dimensions):
            raise ValueError("reference velocities require six-component observations")
        factors = [np.ascontiguousarray(np.asarray(value, dtype=np.float64)) for value in factor_blocks]
        if len(factors) != rows:
            raise ValueError("factor_blocks must contain one factor per observation")
        for index, (factor, dimension) in enumerate(zip(factors, self.dimensions, strict=True)):
            if factor.shape != (dimension, dimension) or not np.all(np.isfinite(factor)):
                raise ValueError(f"factor block {index} must have shape ({dimension}, {dimension}) and be finite")
            if np.any(np.diag(factor) <= 0.0):
                raise ValueError(f"factor block {index} must have positive diagonals")
        if not np.isfinite(float(mu_km3_s2)) or float(mu_km3_s2) <= 0.0:
            raise ValueError("mu_km3_s2 must be finite and positive")
        self.step_widths_s = np.ascontiguousarray(np.asarray(step_widths_s, dtype=np.float64)).reshape(-1)
        self.sample_end_indices = [int(value) for value in sample_end_indices]
        if not np.all(np.isfinite(self.step_widths_s)) or np.any(self.step_widths_s <= 0.0):
            raise ValueError("step widths must be finite and positive")
        if len(self.sample_end_indices) != rows or any(
            value < 0 or value > self.step_widths_s.size for value in self.sample_end_indices
        ) or any(
            right < left
            for left, right in zip(
                self.sample_end_indices[:-1], self.sample_end_indices[1:], strict=True
            )
        ):
            raise ValueError("sample_end_indices must be non-decreasing and within the step schedule")
        if (self.sample_end_indices[-1] if self.sample_end_indices else 0) != self.step_widths_s.size:
            raise ValueError("step schedule must be consumed by the final observation epoch")
        schedule_time = 0.0
        previous_end = 0
        for target, end in zip(self.times_s, self.sample_end_indices, strict=True):
            schedule_time += float(np.sum(self.step_widths_s[previous_end:end], dtype=np.float64))
            if not np.isclose(schedule_time, target, rtol=1.0e-12, atol=1.0e-9):
                raise ValueError("step schedule does not terminate at each observation epoch")
            previous_end = end
        self.factors = factors
        self._native = native_class(
            self.times_s.tolist(),
            self.positions.reshape(-1).tolist(),
            self.velocities.reshape(-1).tolist(),
            np.concatenate([factor.reshape(-1) for factor in factors]).tolist(),
            self.dimensions,
            float(mu_km3_s2),
            bool(include_j2),
            self.step_widths_s.tolist(),
            self.sample_end_indices,
        )

    def evaluate(self, initial_state: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.ascontiguousarray(np.asarray(initial_state, dtype=np.float64)).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain six finite values")
        bytes_function = getattr(self._native, "evaluate_bytes", None)
        residual_size = int(sum(self.dimensions))
        if bytes_function is not None:
            raw_states, raw_residuals = bytes_function(_f64_bytes(state))
            return (
                _f64_from_bytes(raw_states, (self.times_s.size, 6)),
                _f64_from_bytes(raw_residuals, (residual_size,)),
            )
        raw_states, raw_residuals = self._native.evaluate(state.tolist())
        return (
            np.asarray(raw_states, dtype=np.float64).reshape(self.times_s.size, 6),
            np.asarray(raw_residuals, dtype=np.float64).reshape(residual_size),
        )


class RustONPForcePlanObservationContext:
    """Evaluate a passive OD arc through the configured native force plan.

    The validated candidate owns the force configuration and environment. The
    exact scheduler widths come from the ordinary SimulationSession. Only the
    supported built-in fixed or adaptive force plan is batched; callers retain the ordinary
    session path for every other configuration.
    """

    backend_name = "rust_onp_force_plan_observation_context"

    def __init__(
        self,
        artifact,
        *,
        object_id: str,
        times_s: Sequence[float] | np.ndarray,
        reference_positions_km: Sequence[Sequence[float]] | np.ndarray,
        reference_velocities_km_s: Sequence[Sequence[float]] | np.ndarray | None,
        factor_blocks: Sequence[np.ndarray],
        step_widths_s: Sequence[float] | np.ndarray,
        sample_end_indices: Sequence[int],
        guard_physical_steps: bool = False,
    ) -> None:
        from sim.api import SimulationSession
        from sim.dynamics.model import OrbitalAttitudeDynamics
        from sim.dynamics.orbit.accelerations import OrbitContext
        from sim.dynamics.orbit.propagator import OrbitPropagator

        self._artifact = artifact
        self.guard_physical_steps = bool(guard_physical_steps)
        self._object_id = str(object_id)
        self._parameter_contexts = OrderedDict()
        self.times_s = np.asarray(times_s, dtype=np.float64).reshape(-1).copy()
        self.positions = np.asarray(reference_positions_km, dtype=np.float64)
        self.velocities = (
            None if reference_velocities_km_s is None
            else np.asarray(reference_velocities_km_s, dtype=np.float64)
        )
        self.factors = [np.asarray(value, dtype=np.float64) for value in factor_blocks]
        self.widths = np.asarray(step_widths_s, dtype=np.float64).reshape(-1)
        self.ends = np.asarray(sample_end_indices, dtype=np.int64).reshape(-1)
        rows = self.times_s.size
        dimension = 6 if self.velocities is not None else 3
        if rows == 0 or not np.all(np.isfinite(self.times_s)) or self.times_s[0] < 0.0 or np.any(np.diff(self.times_s) < 0.0):
            raise ValueError("observation epochs must be finite, non-negative, and non-decreasing")
        if self.positions.shape != (rows, 3) or not np.all(np.isfinite(self.positions)):
            raise ValueError("reference positions must be finite (rows, 3) values")
        if self.velocities is not None and (self.velocities.shape != (rows, 3) or not np.all(np.isfinite(self.velocities))):
            raise ValueError("reference velocities must be finite (rows, 3) values")
        if len(self.factors) != rows or any(factor.shape != (dimension, dimension) for factor in self.factors):
            raise ValueError("observation factors must match the observation rows")
        if self.ends.size != rows or np.any(np.diff(self.ends) < 0) or self.ends[0] < 0 or self.ends[-1] != self.widths.size:
            raise ValueError("sample ends must match the complete step schedule")
        if not np.all(np.isfinite(self.widths)) or np.any(self.widths <= 0.0):
            raise ValueError("step widths must be finite and positive")
        if any(not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0.0) for factor in self.factors):
            raise ValueError("observation factors must be finite with positive diagonals")
        end_times = np.concatenate((np.array([0.0]), np.cumsum(self.widths, dtype=np.float64)))
        if not np.allclose(end_times[self.ends], self.times_s, rtol=1.0e-12, atol=1.0e-9):
            raise ValueError("step schedule does not terminate at each observation epoch")

        raw = artifact.to_dict()
        obj = raw.get("objects", {}).get(str(object_id), {})
        flight_software = obj.get("flight_software") or {}
        if flight_software.get("stack") != "fsw.passive":
            raise ValueError("batched force-plan OD requires the passive flight-software stack")
        session = SimulationSession.from_config(artifact, history_mode="dynamic")
        if session.reset() is None:
            raise ValueError("batched force-plan OD requires a single-run candidate")
        engine = session._engine
        agent = engine.agents.get(str(object_id))
        if agent is None or not agent.active or agent.bridge is not None:
            raise ValueError("batched force-plan OD requires one active passive satellite")
        if engine.collision_stepper is not None or engine.system_force_stepper is not None:
            raise ValueError("coupled force or collision models require the session evaluator")
        dynamics = agent.dynamics
        if type(dynamics) is not OrbitalAttitudeDynamics or type(dynamics.orbit_propagator) is not OrbitPropagator:
            raise ValueError("batched force-plan OD requires standard ONP dynamics")
        if dynamics.resource_model is not None or dynamics.propagate_attitude and engine.attitude_enabled:
            raise ValueError("attitude or resource dynamics require the session evaluator")
        if dynamics.geometry_area_profile is not None or dynamics.use_rectangular_prism_for_aero_srp:
            raise ValueError("dynamic area geometry requires the session evaluator")
        if dynamics.lift_axis_body is not None and float(dynamics.lift_coefficient) != 0.0:
            raise ValueError("body-axis lift requires the session evaluator")
        self.propagator = dynamics.orbit_propagator
        if self.propagator.numeric_backend != "rust":
            raise ValueError("batched force-plan OD requires Rust orbit propagation")
        self.env = engine.satellite_stepper.passive_history._environment(agent)
        self.ctx = OrbitContext(
            dynamics.mu_km3_s2, agent.truth.mass_kg,
            dynamics.area_m2, dynamics.cd, dynamics.cr,
        )
        self.step_times = np.concatenate((np.array([0.0]), np.cumsum(self.widths[:-1], dtype=np.float64))) if self.widths.size else np.empty(0)
        unique_ends = sorted(set(int(value) for value in self.ends if value > 0))
        if self.guard_physical_steps:
            # Collect existing integration endpoints without adding integration
            # boundaries. This detects an impact between observation epochs.
            self.unique_ends = list(range(1, self.widths.size + 1))
            self.row_indices = self.ends.copy()
        else:
            self.unique_ends = unique_ends
            row_for_end = {0: 0, **{end: index + 1 for index, end in enumerate(unique_ends)}}
            self.row_indices = np.asarray([row_for_end[int(end)] for end in self.ends], dtype=np.int64)

    def evaluate(self, initial_state: Sequence[float] | np.ndarray, *, specs=None) -> tuple[np.ndarray, np.ndarray]:
        from sim.rust_estimation_backend import measurement_whiten_variable_rows

        if specs is not None:
            raw = self._artifact.to_dict()
            original = raw["objects"][self._object_id].get("specs", {})
            key = json.dumps(specs, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if specs != original:
                context = self._parameter_contexts.get(key)
                if context is None:
                    from sim.scenarios import ScenarioArtifact

                    # Reuse the maintained normalization/validation path for each
                    # distinct parameter geometry. State trials then share it.
                    raw["objects"][self._object_id]["specs"] = specs
                    context = type(self)(ScenarioArtifact.from_dict(raw), object_id=self._object_id,
                        times_s=self.times_s, reference_positions_km=self.positions,
                        reference_velocities_km_s=self.velocities, factor_blocks=self.factors,
                        step_widths_s=self.widths, sample_end_indices=self.ends,
                        guard_physical_steps=self.guard_physical_steps)
                    self._parameter_contexts[key] = context
                    if len(self._parameter_contexts) > 8:
                        self._parameter_contexts.popitem(last=False)
                self._parameter_contexts.move_to_end(key)
                return context.evaluate(initial_state)
        state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must contain six finite values")
        # Each candidate starts with fresh adaptive initial-step state. The
        # sampled API carries h_next within this arc without mutating the prop.
        if self.propagator.integrator in {"adaptive", "rkf78", "dopri5"}:
            result = self.propagator.try_propagate_adaptive_sampled_steps(
                state, self.widths, self.unique_ends, 0.0, np.zeros(3),
                self.env, self.ctx, step_times=self.step_times,
            )
            sampled = None if result is None else result[0]
        else:
            sampled = self.propagator.try_propagate_sampled_steps(
                state, self.widths, self.unique_ends, 0.0, np.zeros(3),
                self.env, self.ctx, step_times=self.step_times,
            )
        if sampled is None:
            raise RuntimeError("configured force model has no native sampled-history kernel")
        if self.guard_physical_steps:
            from sim.dynamics.orbit.environment import EARTH_RADIUS_KM

            physical_states = np.asarray(sampled, dtype=np.float64)
            if (not np.all(np.isfinite(physical_states))
                    or np.any(np.linalg.norm(physical_states[:, :3], axis=1) <= EARTH_RADIUS_KM)):
                raise RuntimeError("native physical step requires maintained session event handling")
        states = np.asarray(sampled, dtype=np.float64)[self.row_indices].copy()
        if states.shape != (self.times_s.size, 6) or not np.all(np.isfinite(states)):
            raise RuntimeError("native force-plan history returned invalid states")
        residual = states[:, :3] - self.positions
        if self.velocities is not None:
            residual = np.hstack((residual, states[:, 3:] - self.velocities))
        whitened = measurement_whiten_variable_rows(list(residual), self.factors)
        return states, whitened
