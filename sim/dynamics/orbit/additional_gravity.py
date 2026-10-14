"""Opt-in Earth ONP composition; the ordinary Earth propagator is unchanged."""

from __future__ import annotations

import numpy as np

from .integrators import AdaptiveStepInfo, combine_adaptive_step_info
from .propagator import OrbitPropagator


class EarthAdditionalGravityPropagator(OrbitPropagator):
    def configure_additional_gravity(self, cfg):
        from sim.dynamics.planetary.additional_gravity import prepare_additional_gravity

        self._additional_config = cfg
        self.additional_gravity = prepare_additional_gravity(cfg)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("additional_gravity", None)
        state["_additional_environment_sha256"] = self.additional_gravity.digest
        return state

    def __setstate__(self, state):
        expected = state.pop("_additional_environment_sha256")
        super().__setstate__(state)
        self.configure_additional_gravity(self._additional_config)
        if self.additional_gravity.digest != expected:
            raise ValueError("Additional gravity resources or configuration changed since serialization")

    def _base_plan(self, x, t, env, ctx):
        from .rust_force_plan import make_plan

        self.additional_gravity.verify_resources()
        *_, callback, base = make_plan(self, x, t, env, ctx)
        if base is None:
            raise ValueError("Additional gravity requires the native ONP force context")
        return callback, base

    def acceleration_at(self, *, t_s, x_eci, command_accel_eci_km_s2, env, ctx):
        from sim.rust_orbit_backend import _command, _state

        callback, base = self._base_plan(x_eci, t_s, env, ctx)
        return np.asarray(
            self.additional_gravity.native.acceleration_earth(
                base, callback, _state(x_eci), float(t_s), _command(command_accel_eci_km_s2)
            )
        )

    def propagate(self, x_eci, dt_s, t_s, command_accel_eci_km_s2, env, ctx):
        from sim.rust_orbit_backend import _command, _state

        callback, base = self._base_plan(x_eci, t_s, env, ctx)
        h = self._rkf78_h_next
        if self._rkf78_last_t_s is None or t_s < self._rkf78_last_t_s - 1.0e-12:
            h = None
        result, details = self.additional_gravity.native.propagate_earth(
            base,
            callback,
            _state(x_eci),
            float(t_s),
            float(dt_s),
            _command(command_accel_eci_km_s2),
            self.integrator,
            float(self.adaptive_atol),
            float(self.adaptive_rtol),
            h,
        )
        self.last_numeric_path = "rust_additional_gravity"
        if self.integrator != "rk4" and int(details[2]) > 0:
            method = "rkf78" if self.integrator == "adaptive" else self.integrator
            info = AdaptiveStepInfo(
                method=method,
                accepted_steps=int(details[0]),
                rejected_steps=int(details[1]),
                attempted_steps=int(details[2]),
                min_step_s=details[3],
                max_step_s=details[4],
                final_step_s=details[5],
                suggested_next_step_s=details[6],
                max_error_ratio=details[7],
            )
            self.last_adaptive_step_info = info
            self._rkf78_h_next, self._rkf78_last_t_s = info.suggested_next_step_s, float(t_s + dt_s)
            previous = [] if self.adaptive_step_info is None else [self.adaptive_step_info]
            self.adaptive_step_info = combine_adaptive_step_info(method, [*previous, info])
        return np.asarray(result)

    # Existing speculative history kernels do not contain the additional field.
    # Leave lifecycle stepping in charge for this opt-in propagator.
    def try_propagate_fixed_steps(self, *args, **kwargs):
        return None

    def try_propagate_sampled_steps(self, *args, **kwargs):
        return None

    def try_propagate_adaptive_sampled_steps(self, *args, **kwargs):
        return None

    def propagation_metadata(self):
        return {
            **super().propagation_metadata(),
            "numeric_backend": "rust",
            "additional_gravity": self.additional_gravity.metadata(),
            "speculative_history": False,
        }
