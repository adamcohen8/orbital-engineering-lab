"""Optional deterministic batches of decision-snapshot coast forecasts."""
from __future__ import annotations

import numpy as np


def eligible(dynamics, truth, elapsed, env):
    prop = getattr(dynamics, "orbit_propagator", None)
    if getattr(prop, "numeric_backend", None) != "rust" or elapsed <= 0:
        return False
    from sim.dynamics.model import OrbitalAttitudeDynamics
    from sim.dynamics.orbit.propagator import OrbitPropagator
    from sim.dynamics.orbit.rust_force_plan import _CODES, native_force_plan_codes

    return bool(
        type(dynamics) is OrbitalAttitudeDynamics and type(prop) is OrbitPropagator
        and prop.numeric_backend == "rust" and prop.integrator == "rk4"
        and prop.model == "two_body" and prop.state_frame == "eci"
        and "step" not in vars(dynamics) and "propagate" not in vars(prop)
        and "acceleration_at" not in vars(prop)
        and dynamics.resource_model is None and dynamics.geometry_area_profile is None
        and not dynamics.use_rectangular_prism_for_aero_srp and dynamics.lift_axis_body is None
        and all(any(plugin is known for known in _CODES) for plugin in prop.plugins)
        and native_force_plan_codes(prop, env) is not None
        and np.isfinite(dynamics.mu_km3_s2) and dynamics.mu_km3_s2 > 0
        and elapsed > 0 and np.isfinite(elapsed) and np.isfinite(truth.t_s + elapsed)
        and truth.mass_kg > 0 and np.isfinite(truth.mass_kg)
        and not any(callable(value) for key, value in env.items() if not str(key).startswith("_"))
        and not env.get("_rust_forecast_batch_disabled", False)
    )


def _prepare(dynamics, truth, elapsed, env, native):
    from sim.dynamics.orbit.accelerations import OrbitContext
    from sim.dynamics.orbit.rust_force_plan import _CODES, make_plan, native_force_plan_codes
    from sim.rust_orbit_backend import native_harmonic_degree_limit

    prop = dynamics.orbit_propagator
    codes = [_CODES[plugin] for plugin in prop.plugins]
    if all(code in (6, 7, 8) for code in codes):
        key = (float(dynamics.mu_km3_s2), tuple(codes))
        cached = getattr(prop, "_rust_forecast_zonal_context", None)
        if cached is None or cached[0] != key:
            scalars = [float(dynamics.mu_km3_s2)] + [0.0] * 14
            cached = (key, native.ONPForceContext(codes, scalars, 0, (0, 0), []))
            prop._rust_forecast_zonal_context = cached
        context = cached[1]
        def stage(_time, _state):
            return [1., 0., 0., 0., 1., 0., 0., 0., 1.] + [0.] * 10
    else:
        local_env = dict(env)
        if dynamics.drag_area_m2 is not None and "drag_area_m2" not in local_env:
            local_env["drag_area_m2"] = float(dynamics.drag_area_m2)
        if dynamics.srp_area_m2 is not None:
            local_env["srp_area_m2"] = float(dynamics.srp_area_m2)
        x = np.r_[truth.position_eci_km, truth.velocity_eci_km_s]
        orbit_context = OrbitContext(dynamics.mu_km3_s2, truth.mass_kg,
                                     dynamics.area_m2, dynamics.cd, dynamics.cr)
        if native_force_plan_codes(prop, local_env) is None:
            return None
        if 1 in codes:
            from sim.dynamics.orbit.propagator import spherical_harmonics_plugin
            if local_env.get("_compiled_spherical_harmonics_terms") is None:
                spherical_harmonics_plugin(truth.t_s, x, local_env, orbit_context)
            terms = local_env.get("_compiled_spherical_harmonics_terms")
            if (terms is None or not terms.all_normalized
                    or terms.n_max > native_harmonic_degree_limit()):
                return None
        _, _, _, dims, _, stage, context = make_plan(prop, x, truth.t_s, local_env, orbit_context)
        if context is None or (1 in codes and dims[0] > native_harmonic_degree_limit()):
            return None
    step = dynamics._effective_substep(dynamics.orbit_substep_s, elapsed)
    widths = dynamics._substep_sequence(elapsed, step)
    return context, stage, np.r_[truth.position_eci_km, truth.velocity_eci_km_s].tolist(), float(truth.t_s), widths


def forecast_batch(entries, elapsed, env):
    """Yield (entry, predicted) in order; preserve completed rows on failure.

    Ineligible rows and older wheels use the ordinary dynamics owner. Entries
    contain arbitrary caller metadata followed by dynamics and source truth.
    """
    from sim.core.models import Command
    from sim.rust_orbit_backend import _extension

    native = _extension()
    function = getattr(native, "coast_forecast_batch", None)
    if function is None:
        for entry in entries:
            yield entry, entry[-2].step(state=entry[-1].copy(), command=Command.zero(), env=dict(env), dt_s=elapsed)
        return
    prepared = []
    pending = []

    def flush():
        if not pending:
            return
        contexts, stages, states, starts, widths = zip(*prepared)
        rows, failure = function(list(contexts), list(stages), list(states), list(starts), list(widths))
        for entry, row in zip(pending, rows):
            dynamics, source = entry[-2:]
            predicted = source.copy()
            predicted.position_eci_km = np.array(row[:3], dtype=float)
            predicted.velocity_eci_km_s = np.array(row[3:], dtype=float)
            predicted.resource_state = None
            predicted.mass_kg = max(0.0, float(source.mass_kg))
            predicted.t_s = float(source.t_s + elapsed)
            dynamics.orbit_propagator.last_numeric_path = "rust_native_forecast_batch"
            yield entry, predicted
        if failure is not None:
            raise failure
        pending.clear()
        prepared.clear()

    for entry in entries:
        try:
            row = _prepare(entry[-2], entry[-1], elapsed, env, native)
        except Exception:
            # Earlier serial rows must finish before this preparation error.
            yield from flush()
            raise
        if row is None:
            yield from flush()
            yield entry, entry[-2].step(state=entry[-1].copy(), command=Command.zero(), env=dict(env), dt_s=elapsed)
        else:
            pending.append(entry)
            prepared.append(row)
    yield from flush()
