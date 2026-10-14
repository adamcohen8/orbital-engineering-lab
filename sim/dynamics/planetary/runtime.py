"""Passive runtime adapter for body-centered J2000 trajectories.

StateTruth's legacy internal field names remain an engine compatibility seam.
Public planetary evidence uses explicitly neutral coordinates and origin IDs.
"""

from __future__ import annotations

import numpy as np

from sim.core.models import StateTruth

from .propagation import PlanetaryForcePlan, PlanetaryPropagator


def initial_truth(obj, environment):
    initial = obj.initial_state
    if "position_inertial_km" in initial:
        r = np.asarray(initial["position_inertial_km"], dtype=float)
        v = np.asarray(initial["velocity_inertial_km_s"], dtype=float)
    else:
        from sim.dynamics.orbit.elements import coe_to_rv_eci

        coes = initial["coes"]
        r, v = coe_to_rv_eci(
            a_km=coes.get("a_km", coes.get("semi_major_axis_km")),
            ecc=coes.get("ecc", coes.get("e", 0.0)),
            inc_deg=coes.get("inc_deg", coes.get("inclination_deg", 0.0)),
            raan_deg=coes.get("raan_deg", 0.0),
            argp_deg=coes.get("argp_deg", coes.get("arg_periapsis_deg", 0.0)),
            true_anomaly_deg=coes.get("ta_deg", coes.get("true_anomaly_deg", 0.0)),
            mu_km3_s2=environment.gravity.mu_km3_s2,
        )
    if r.shape != (3,) or v.shape != (3,) or not np.all(np.isfinite(np.r_[r, v])):
        raise ValueError(
            "Planetary initial Cartesian state must contain finite three-vectors"
        )
    if np.linalg.norm(r) <= 0.0:
        raise ValueError("Planetary initial radius must be positive")
    impact = environment.selections["impact"]
    if impact["enabled"] and np.linalg.norm(r) <= impact.get(
        "radius_km", environment.body.reference_radius_km
    ):
        raise ValueError(
            "Planetary initial state is inside the selected impact surface"
        )
    return StateTruth(
        r.copy(),
        v.copy(),
        np.array(initial.get("attitude_quat_bn", [1.0, 0.0, 0.0, 0.0]), dtype=float),
        np.array(initial.get("angular_rate_body_rad_s", [0.0, 0.0, 0.0]), dtype=float),
        float(obj.specs["mass_kg"]),
        0.0,
    )


class PlanetaryDynamics:
    propagate_attitude = False
    resource_model = None

    def __init__(self, environment, obj, settings):
        self.environment = environment
        self.mu_km3_s2 = environment.gravity.mu_km3_s2
        specs = obj.specs
        plan = PlanetaryForcePlan(
            environment,
            mass_kg=specs["mass_kg"],
            cd=specs.get("cd", 0.0),
            drag_area_m2=specs.get("drag_area_m2", 0.0),
            cr=specs.get("cr", 0.0),
            srp_area_m2=specs.get("srp_area_m2", 0.0),
            numeric_backend=settings.get("numeric_backend", "rust"),
        )
        self.orbit_propagator = PlanetaryPropagator(plan, settings)
        from .native import module

        self._gravity_domain_error = module().PlanetaryGravityDomainError
        self.gravity_domain_exit = None
        self._step_start_s = None
        self._adaptive_info_before_step = None

    def step(self, state, command, env, dt_s):
        if (
            np.any(command.thrust_eci_km_s2)
            or np.any(command.torque_body_nm)
            or command.mode_flags
        ):
            raise ValueError(
                "Planetary passive dynamics cannot consume physical commands"
            )
        self.environment.verify_resources()
        self.gravity_domain_exit = None
        self._step_start_s = state.t_s
        self._adaptive_info_before_step = self.orbit_propagator.last_adaptive_info
        result = state.copy()
        try:
            x = self.orbit_propagator.propagate(
                np.r_[state.position_eci_km, state.velocity_eci_km_s], dt_s, state.t_s
            )
            # RK4's weighted endpoint is not itself an integration stage.
            # Validate it before accepting or recording the resulting state.
            plan = self.orbit_propagator.plan
            plan.stage(state.t_s + dt_s, x)
        except self._gravity_domain_error:
            self.gravity_domain_exit = {
                "last_valid_sample_time_s": float(state.t_s),
                "attempted_end_time_s": float(state.t_s + dt_s),
                "gravity_reference_radius_km": self.environment.gravity.radius_km,
                "physical_impact_recorded": False,
            }
            raise
        if not np.all(np.isfinite(x)):
            raise ValueError("Planetary integration produced a nonfinite state")
        result.position_eci_km = x[:3]
        result.velocity_eci_km_s = x[3:]
        result.t_s = state.t_s + dt_s
        return result


def create_runtime(object_id, obj, cfg, environment=None):
    from sim.runtime.models import AgentRuntime

    from .preparation import prepare_environment

    environment = environment or prepare_environment(cfg)
    environment.verify_resources()
    return AgentRuntime(
        object_id=object_id,
        kind="satellite",
        enabled=True,
        active=True,
        truth=initial_truth(obj, environment),
        belief=None,
        sensor=None,
        estimator=None,
        orbit_controller=None,
        attitude_controller=None,
        dynamics=PlanetaryDynamics(
            environment, obj, dict(cfg.simulator.dynamics.get("orbit", {}))
        ),
        knowledge_base=None,
        bridge=None,
        mission_strategy=None,
        mission_execution=None,
        rocket_sim=None,
        rocket_state=None,
        rocket_guidance=None,
        deploy_source=None,
        deploy_time_s=None,
        deploy_dv_body_m_s=None,
        initialization_delay_s=0.0,
        control_available_time_s=0.0,
        mission_modules=[],
        waiting_for_launch=False,
        runtime_profile="trajectory_only",
    )


class PlanetaryTerminationMonitor:
    def __init__(self, engine):
        self.engine = engine

    def check_reentry(self, **kwargs):
        return False

    def update_rocket_insertion(self, **kwargs):
        return None

    def handle_gravity_domain_exit(self, error, inputs):
        from .native import module

        if not isinstance(error, module().PlanetaryGravityDomainError):
            return False
        failed = next(
            (item for item in inputs if item.agent.dynamics.gravity_domain_exit is not None),
            None,
        )
        if failed is None:
            return False
        # Serial stepping may already have advanced another object in this
        # interval. Restore every object to the recorded interval-start state.
        for item in inputs:
            item.agent.truth = item.initial_truth.copy()
            dynamics = item.agent.dynamics
            if dynamics._step_start_s == item.t_s:
                dynamics.orbit_propagator.last_adaptive_info = dynamics._adaptive_info_before_step
        e = self.engine
        e.terminated_early = True
        e.termination_reason = "primary_gravity_domain_exit"
        e.termination_time_s = float(failed.t_s)
        e.termination_object_id = failed.object_id
        e.planetary_domain_exit = {
            **failed.agent.dynamics.gravity_domain_exit,
            "object_id": failed.object_id,
        }
        return True

    def check_earth_impact(self, *, t_s, include_rockets=True):
        e = self.engine
        selection = e.planetary_environment.selections["impact"]
        if not selection["enabled"]:
            return False
        radius = selection.get(
            "radius_km", e.planetary_environment.body.reference_radius_km
        )
        for oid, agent in e.agents.items():
            if agent.active and np.linalg.norm(agent.truth.position_eci_km) <= radius:
                e.terminated_early = True
                e.termination_reason = "central_body_impact"
                e.termination_time_s = float(t_s)
                e.termination_object_id = oid
                return True
        return False
