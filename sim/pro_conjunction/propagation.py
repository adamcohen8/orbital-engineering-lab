"""ONP histories and shared-OD covariance transitions, including an impulse map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.analysis.conjunction_geometry import StateHistory
from sim.analysis.conjunction_probability import ric_basis
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, j2_plugin
from sim.estimation.augmented_dynamics import AugmentedONPDynamics


def dynamics(problem, obj):
    prop = OrbitPropagator(
        integrator="rk4", numeric_backend="python", plugins=[j2_plugin] if problem["force_model"] == "j2" else []
    )
    return AugmentedONPDynamics(
        propagator=prop,
        context=OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=100),
        environment={},
        step_s=problem["step_s"],
        acceleration_spectral_density=obj["acceleration_spectral_density_km2_s3"],
    )


def impulse(state, burn):
    """Same instantaneous ECI/RIC impulse convention as public targeting."""
    x = np.asarray(state, dtype=float).copy()
    basis = ric_basis(x) if burn["frame"] == "ric" else np.eye(3)
    x[3:] += basis @ np.asarray(burn["delta_v_m_s"]) / 1000
    return x


def impulse_covariance(state, cov, burn):
    # Differentiate the local RIC steering map, including state-dependent axes.
    jac = np.eye(6)
    if burn["frame"] == "ric":
        for i, h in enumerate([1e-3] * 3 + [1e-6] * 3):
            plus, minus = state.copy(), state.copy()
            plus[i] += h
            minus[i] -= h
            jac[:, i] = (impulse(plus, burn) - impulse(minus, burn)) / (2 * h)
    basis = ric_basis(state) if burn["frame"] == "ric" else np.eye(3)
    g = np.vstack((np.zeros((3, 3)), basis / 1000))
    out = jac @ cov @ jac.T + g @ np.asarray(burn["execution_covariance_m2_s2"]) @ g.T
    return impulse(state, burn), 0.5 * (out + out.T), jac


@dataclass
class UncertainHistory:
    history: StateHistory
    covariances: list
    incoming_covariances: list
    dynamics: AugmentedONPDynamics
    burn_jacobian: list | None

    def at(self, time_s, *, side="right"):
        if side not in {"left", "right"}:
            raise ValueError("covariance history side must be 'left' or 'right'")
        ts = np.asarray(self.history.times_s)
        i = max(0, int(np.searchsorted(ts, time_s, side="right")) - 1)
        if time_s < ts[0] or time_s > ts[-1]:
            raise ValueError("covariance extrapolation is forbidden")
        state = np.array(self.history.states_eci_km_km_s[i])
        cov = np.array(self.covariances[i])
        if abs(time_s - ts[i]) < 1e-10:
            if side == "left":
                state[3:] = self.history.incoming_velocities_eci_km_s[i]
                cov = np.array(self.incoming_covariances[i])
            return state, cov
        x, phi, q = self.dynamics.predict(state, float(ts[i]), time_s - ts[i])
        return x, self.dynamics.stabilize(phi @ cov @ phi.T + q)


def propagate(problem, obj, *, burn=None, covariance=True, state=None):
    dyn = dynamics(problem, obj)
    x = np.array(obj["state_eci_km_km_s"] if state is None else state, dtype=float)
    p = np.array(obj["uncertainty"]["covariance"])
    ts, xs, ps, incoming, incoming_ps = [], [], [], [], []
    t, jac = 0.0, None
    while True:
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[:3]) <= EARTH_RADIUS_KM:
            raise ValueError("nonfinite or Earth-intersecting history outside conjunction model envelope")
        pre, pre_p = x.copy(), p.copy()
        if burn is not None and abs(t - burn["time_s"]) < 1e-9:
            if covariance:
                x, p, j = impulse_covariance(x, p, burn)
                jac = j.tolist()
            else:
                x = impulse(x, burn)
        ts.append(t)
        xs.append(x.copy())
        incoming.append(pre[3:].copy())
        ps.append(p.tolist())
        incoming_ps.append(pre_p.tolist())
        if t >= problem["duration_s"]:
            break
        end = min(t + problem["step_s"], problem["duration_s"])
        if burn is not None and t < burn["time_s"] < end:
            end = burn["time_s"]
        if covariance:
            x, phi, q = dyn.predict(x, t, end - t)
            p = dyn.stabilize(phi @ p @ phi.T + q)
        else:
            x = dyn.propagate_mean(x, t, end - t)
        t = end
    h = StateHistory.from_arrays(ts, xs, incoming_velocities_eci_km_s=incoming)
    return UncertainHistory(h, ps, incoming_ps, dyn, jac)
