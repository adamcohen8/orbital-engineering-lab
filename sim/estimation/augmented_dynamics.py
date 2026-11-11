"""Augmented ONP transitions for the shared sequential OD estimator.

Physics remains in OrbitPropagator. Sensitivities differentiate that owner;
continuous noise is discretized locally with Van Loan's block exponential.
Noise discretization is refined with the integration grid, not measurement cadence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.linalg import expm

from sim.estimation.covariance import validate_covariance

ORBIT_NAMES = ("x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s")


@dataclass(frozen=True)
class AugmentedParameter:
    name: str
    kind: str
    unit: str
    step: float
    model: str = "constant"
    spectral_density: float = 0.0
    mean: float = 0.0
    correlation_time_s: float | None = None
    station_id: str | None = None
    estimate: bool = True

    def __post_init__(self):
        units = {"cd_scale": "1", "srp_scale": "1", "range_bias": "km", "clock_offset": "s"}
        if (
            self.kind not in units
            or self.unit != units[self.kind]
            or not isinstance(self.name, str)
            or not self.name.strip()
        ):
            raise ValueError("declare a supported parameter kind, name and native unit")
        if self.model not in {"constant", "random_walk", "gauss_markov"}:
            raise ValueError("parameter model must be constant, random_walk or gauss_markov")
        if (
            not np.all(np.isfinite([self.step, self.spectral_density, self.mean]))
            or self.step <= 0
            or self.spectral_density < 0
        ):
            raise ValueError("parameter step must be positive and spectral density nonnegative")
        if self.model == "constant" and self.spectral_density != 0:
            raise ValueError("constant parameters cannot have process noise")
        if self.model == "gauss_markov":
            if (
                self.correlation_time_s is None
                or not np.isfinite(self.correlation_time_s)
                or self.correlation_time_s <= 0
            ):
                raise ValueError("Gauss-Markov parameters require positive correlation_time_s")
        elif self.correlation_time_s is not None:
            raise ValueError("correlation time applies only to Gauss-Markov parameters")
        if not isinstance(self.estimate, bool):
            raise ValueError("estimate must be boolean")
        if self.kind in {"range_bias", "clock_offset"} and not self.station_id:
            raise ValueError("sensor parameters require an explicit station_id")

    @property
    def rate(self):
        return 1.0 / self.correlation_time_s if self.model == "gauss_markov" else 0.0


def discrete_noise(generator, spectral_density, dt_s):
    """Exact integral for this locally frozen linear generator, including cross terms."""
    a, q = np.asarray(generator), np.asarray(spectral_density)
    n = len(a)
    if a.shape != (n, n) or not np.all(np.isfinite(a)) or not np.isfinite(dt_s) or dt_s < 0:
        raise ValueError("invalid continuous-noise generator or interval")
    validate_covariance(q, n, name="spectral density")
    block = np.block([[a, q], [np.zeros_like(a), -a.T]]) * dt_s
    e = expm(block)
    result = e[:n, n:] @ e[:n, :n].T
    return 0.5 * (result + result.T)


class AugmentedONPDynamics:
    """Fixed named slots; inactive parameters remain correlated consider states.

    Events change estimation masks/noise statistics at explicit integration
    boundaries. Slots are retained so cross-covariance is never discarded.
    Smoothing stops across a statistics/mask change (a declared segment boundary).
    """

    def __init__(
        self,
        *,
        propagator,
        context,
        environment,
        parameters=(),
        step_s=20.0,
        acceleration_spectral_density=(0.0, 0.0, 0.0),
        model_identity=None,
        events=(),
    ):
        self.propagator, self.context = propagator, context
        self.environment = copy.deepcopy(dict(environment))
        self.parameters = tuple(parameters)
        self.step_s = float(step_s)
        self.events = copy.deepcopy(list(events))
        self.spectral_density = np.asarray(acceleration_spectral_density, dtype=float)
        self.model_identity = copy.deepcopy(model_identity or {})
        self.state_names = (*ORBIT_NAMES, *(p.name for p in self.parameters))
        self.state_units = (*(["km"] * 3), *(["km/s"] * 3), *(p.unit for p in self.parameters))
        self.steps = np.array([1e-3] * 3 + [1e-6] * 3 + [p.step for p in self.parameters])
        if len(set(self.state_names)) != len(self.state_names) or len(
            {(p.kind, p.station_id) for p in self.parameters}
        ) != len(self.parameters):
            raise ValueError("parameter names and kind/station bindings must be unique")
        if self.propagator.state_frame != "eci" or not np.isfinite(self.step_s) or self.step_s <= 0:
            raise ValueError("augmented OD requires ECI ONP and a positive finite step")
        if (
            self.spectral_density.shape != (3,)
            or np.any(~np.isfinite(self.spectral_density))
            or np.any(self.spectral_density < 0)
        ):
            raise ValueError("three nonnegative acceleration spectral densities are required")
        for event in self.events:
            if set(event) != {"time_s", "parameters"} or not np.isfinite(event["time_s"]) or event["time_s"] <= 0:
                raise ValueError("events require a positive epoch and parameter changes")
            for name, changes in event["parameters"].items():
                if name not in self.state_names[6:] or set(changes) - {
                    "estimate",
                    "spectral_density",
                    "model",
                    "correlation_time_s",
                    "mean",
                }:
                    raise ValueError("events may change only declared parameter masks/statistics")
        if any(a["time_s"] >= b["time_s"] for a, b in zip(self.events, self.events[1:])):
            raise ValueError("event epochs must be strictly increasing")
        # Validate every resulting parameter configuration before executing.
        for event in self.events:
            self.parameters_at(event["time_s"])

    def identity(self):
        return {
            "owner": __name__,
            "state_names": list(self.state_names),
            "parameters": [asdict(p) for p in self.parameters],
            "events": self.events,
            "step_s": self.step_s,
            "acceleration_spectral_density": self.spectral_density.tolist(),
            "noise_model": "substep_local_continuous_Van_Loan",
            "deterministic_adaptive_sensitivity": "whole_interval_for_constant_zero_noise_states",
            "model": self.model_identity,
        }

    def parameters_at(self, t):
        params = list(self.parameters)
        for event in self.events:
            if event["time_s"] <= t:
                params = [replace(p, **event["parameters"].get(p.name, {})) for p in params]
        return params

    def update_mask(self, t):
        return np.array([True] * 6 + [p.estimate for p in self.parameters_at(t)])

    def boundary_at(self, t, previous_t):
        return any(previous_t < e["time_s"] <= t for e in self.events)

    def stabilize(self, covariance):
        # Preserve an already positive-definite covariance exactly. Repeated
        # eigendecomposition in fixed engineering units loses tiny orbital
        # directions beside broad, inactive station-bias priors on long arcs.
        result = np.asarray(covariance, dtype=float)
        result = 0.5 * (result + result.T)
        if not np.all(np.isfinite(result)) or np.any(np.diag(result) < 0):
            raise ValueError("augmented covariance has invalid variances")
        positive = np.diag(result) > 0
        if np.any(result[~positive, :] != 0) or np.any(result[:, ~positive] != 0):
            raise ValueError("augmented covariance couples a zero-variance coordinate")
        if not np.any(positive):
            return result
        scales = np.sqrt(np.diag(result)[positive])
        normalized = result[np.ix_(positive, positive)] / np.outer(scales, scales)
        try:
            np.linalg.cholesky(normalized)
            return result
        except np.linalg.LinAlgError:
            eig, vectors = np.linalg.eigh(normalized)
        if eig[0] < -1e-10 * len(scales):
            raise ValueError("augmented covariance lost positive semidefiniteness")
        result[np.ix_(positive, positive)] = (
            (vectors * np.maximum(eig, 0.0)) @ vectors.T
        ) * np.outer(scales, scales)
        return 0.5 * (result + result.T)

    def force_inputs(self, state):
        ctx, env = self.context, dict(self.environment)
        for p, value in zip(self.parameters, state[6:], strict=True):
            if p.kind in {"cd_scale", "srp_scale"}:
                if value <= 0:
                    raise ValueError("estimated physical force scales must remain positive")
                if p.kind == "cd_scale":
                    ctx = replace(ctx, cd=self.context.cd * value)
                    env["drag_coefficient"] = ctx.cd
                else:
                    ctx = replace(ctx, cr=self.context.cr * value)
        return ctx, env

    def _map(self, state, t, h, parameters):
        middle, end = state.copy(), state.copy()
        for i, p in enumerate(parameters, 6):
            middle[i] = p.mean + (state[i] - p.mean) * np.exp(-p.rate * h / 2)
            end[i] = p.mean + (state[i] - p.mean) * np.exp(-p.rate * h)
        ctx, env = self.force_inputs(middle)
        # Adaptive step history must not leak from a preceding sensitivity trial.
        prop = copy.copy(self.propagator)
        prop._rkf78_h_next = prop._rkf78_last_t_s = None
        end[:6] = prop.propagate(
            x_eci=state[:6].copy(), dt_s=h, t_s=t, command_accel_eci_km_s2=np.zeros(3), env=env, ctx=ctx
        )
        return end

    def _generator(self, state, t, parameters):
        n = len(state)
        a, q = np.zeros((n, n)), np.zeros((n, n))
        a[:3, 3:6] = np.eye(3)
        q[3:6, 3:6] = np.diag(self.spectral_density)
        for i, step in enumerate(self.steps):
            # Measurement-only states cannot change the acceleration.
            if i >= 6 and parameters[i - 6].kind not in {"cd_scale", "srp_scale"}:
                continue
            accelerations = []
            for sign in (1, -1):
                trial = state.copy()
                trial[i] += sign * step
                ctx, env = self.force_inputs(trial)
                accelerations.append(
                    self.propagator.acceleration_at(
                        t_s=t, x_eci=trial[:6], command_accel_eci_km_s2=np.zeros(3), env=env, ctx=ctx
                    )
                )
            a[3:6, i] = (accelerations[0] - accelerations[1]) / (2 * step)
        for i, p in enumerate(parameters, 6):
            a[i, i], q[i, i] = -p.rate, p.spectral_density
        return a, q

    def predict(self, state, time_s, dt_s):
        x = np.asarray(state, dtype=float).copy()
        if (
            x.shape != (len(self.state_names),)
            or not np.all(np.isfinite(x))
            or not np.isfinite(time_s + dt_s)
            or dt_s < 0
        ):
            raise ValueError("invalid augmented prediction state or interval")
        n = len(x)
        transition, noise = np.eye(n), np.zeros((n, n))
        t, stop = float(time_s), float(time_s + dt_s)
        while t < stop - 1e-10:
            boundary = min([stop, *[e["time_s"] for e in self.events if e["time_s"] > t + 1e-10]])
            parameters = self.parameters_at(t)
            # With constant parameters and no continuous noise, one adaptive
            # interval gives the same maintained dynamics without artificial
            # restart boundaries. Stochastic or time-varying states retain
            # the declared covariance integration grid.
            adaptive_interval = (
                self.propagator.integrator in {"adaptive", "rkf78", "dopri5"}
                and not np.any(self.spectral_density)
                and all(p.rate == 0 and p.spectral_density == 0 for p in parameters)
            )
            h = boundary - t if adaptive_interval else min(self.step_s, boundary - t)
            nominal = self._map(x, t, h, parameters)
            phi = np.empty((n, n))
            for i, step in enumerate(self.steps):
                plus, minus = x.copy(), x.copy()
                plus[i] += step
                minus[i] -= step
                phi[:, i] = (self._map(plus, t, h, parameters) - self._map(minus, t, h, parameters)) / (2 * step)
            if np.any(self.spectral_density) or any(p.spectral_density for p in parameters):
                midpoint = self._map(x, t, h / 2, parameters)
                a, q = self._generator(midpoint, t + h / 2, parameters)
                local_noise = discrete_noise(a, q, h)
            else:
                local_noise = np.zeros_like(noise)
            noise = phi @ noise @ phi.T + local_noise
            transition, x, t = phi @ transition, nominal, t + h
        return x, transition, self.stabilize(noise)

    def propagate_mean(self, state, time_s, dt_s):
        """Same ONP mean as prediction, without unnecessary covariance work."""
        x, t, stop = np.asarray(state, dtype=float).copy(), float(time_s), float(time_s + dt_s)
        if dt_s < 0 or not np.isfinite(stop):
            raise ValueError("mean propagation requires an ordered finite interval")
        while t < stop - 1e-10:
            boundary = min([stop, *[e["time_s"] for e in self.events if e["time_s"] > t + 1e-10]])
            h = min(self.step_s, boundary - t)
            x = self._map(x, t, h, self.parameters_at(t))
            t += h
        return x


def canonical_digest(value: Any):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
