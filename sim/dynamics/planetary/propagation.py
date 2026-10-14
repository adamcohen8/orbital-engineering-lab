"""Runtime facade over the separate prepared Rust planetary plan."""

from __future__ import annotations

import numpy as np

from sim.dynamics.orbit.integrators import AdaptiveStepInfo


class PlanetaryForcePlan:
    numeric_backend = "rust"

    def __init__(
        self,
        environment,
        *,
        mass_kg,
        cd=0.0,
        drag_area_m2=0.0,
        cr=0.0,
        srp_area_m2=0.0,
        numeric_backend="rust",
    ):
        if numeric_backend != "rust":
            raise ValueError(
                "Planetary ONP uses the Rust backend; select numeric_backend=rust"
            )
        self.environment = environment
        self.mass_kg, self.cd, self.drag_area_m2, self.cr, self.srp_area_m2 = map(
            float, (mass_kg, cd, drag_area_m2, cr, srp_area_m2)
        )
        from .native import prepare_native_plan

        self.native = prepare_native_plan(self)

    def stage(self, t_s, x):
        total, diagnostics, components = self.native.stage(
            float(t_s), np.asarray(x, dtype=float).tolist()
        )
        names = ["point_mass", "harmonics", "drag", "srp"]
        names += [
            f"third_body_{b['target_center_id']}"
            for b in self.environment.selections["third_bodies"]
        ]
        if self.environment.additional_gravity is not None:
            names += [f"additional_gravity_{b['target_center_id']}" for b in self.environment.additional_gravity.sources]
        return np.asarray(total), {
            "altitude_km": diagnostics[0],
            "density_kg_m3": diagnostics[1],
            "relative_speed_m_s": diagnostics[2],
            "within_atmosphere_domain": bool(diagnostics[3]),
            "forces_km_s2": dict(zip(names, components)),
            "evaluation_backend": "rust",
        }


class PlanetaryPropagator:
    state_frame = "planetary_inertial"
    model = "two_body"
    numeric_backend = "rust"

    def __init__(self, plan, settings):
        self.plan = plan
        self.integrator = settings.get("integrator", "rk4")
        self.atol, self.rtol = settings.get("adaptive_atol", 1.0e-10), settings.get(
            "adaptive_rtol", 1.0e-10
        )
        self.last_adaptive_info = None
        self.substep_s = settings.get("orbit_substep_s")

    def propagate(self, x, dt_s, t_s):
        if self.substep_s is not None and dt_s > self.substep_s:
            remaining = float(dt_s)
            state = np.asarray(x, dtype=float)
            time = float(t_s)
            while remaining > 1e-10:
                step = min(float(self.substep_s), remaining)
                state = self._propagate_segment(state, step, time)
                time += step
                remaining -= step
            return state
        return self._propagate_segment(x, dt_s, t_s)

    def _propagate_segment(self, x, dt_s, t_s):
        result, info = self.plan.native.propagate(
            np.asarray(x, dtype=float).tolist(),
            float(t_s),
            float(dt_s),
            self.integrator,
            float(self.atol),
            float(self.rtol),
        )
        if self.integrator != "rk4":
            self.last_adaptive_info = AdaptiveStepInfo(
                method=self.integrator,
                accepted_steps=int(info[0]),
                rejected_steps=int(info[1]),
                attempted_steps=int(info[2]),
                min_step_s=info[3],
                max_step_s=info[4],
                final_step_s=info[5],
                suggested_next_step_s=info[6],
                max_error_ratio=info[7],
            )
        return np.asarray(result)

    def propagation_metadata(self):
        return {
            "propagation_method": "special",
            "propagator_family": "ONP",
            "propagator_name": "planetary",
            "state_history_frame": self.state_frame,
            "output_frame": "J2000",
            "native_frame": "J2000",
            "central_body": self.plan.environment.body.body_id,
            "origin_center_id": self.plan.environment.body.center_id,
            "numeric_backend": "rust",
            "integrator": self.integrator,
            "stage_callbacks": False,
            "environment": self.plan.environment.metadata(),
            "speculative_history": False,
        }
