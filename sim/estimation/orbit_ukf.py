from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.accelerations import OrbitContext
from sim.dynamics.orbit.propagator import OrbitPropagator
from sim.numeric_backend import normalize_numeric_backend

rust_estimation_backend = None
_RUST_ESTIMATION_BACKEND_LOCK = RLock()


def _load_rust_estimation_backend():
    global rust_estimation_backend
    backend = rust_estimation_backend
    if backend is None:
        with _RUST_ESTIMATION_BACKEND_LOCK:
            backend = rust_estimation_backend
            if backend is None:
                from sim import rust_estimation_backend as loaded_backend

                backend = loaded_backend
                rust_estimation_backend = backend
    return backend


@dataclass
class OrbitUKFEstimator(Estimator):
    propagator: OrbitPropagator
    context: OrbitContext
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    numeric_backend: str = "rust"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be finite and positive.")
        if not np.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be finite and positive.")
        if not np.isfinite(float(self.beta)) or not np.isfinite(float(self.kappa)):
            raise ValueError("beta and kappa must be finite.")
        self.numeric_backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
        if self.numeric_backend not in {"python", "rust"}:
            raise ValueError("numeric_backend must be 'python' or 'rust'.")
        for name in ("process_noise_diag", "meas_noise_diag"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite, nonnegative values.")
            setattr(self, name, values)

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        belief_t_s = float(belief.last_update_t_s)
        if not np.isfinite(output_t_s) or not np.isfinite(belief_t_s) or output_t_s < belief_t_s:
            raise ValueError("output epoch must be finite and not precede the belief epoch.")
        state = np.asarray(belief.state, dtype=float).reshape(-1)
        covariance = np.asarray(belief.covariance, dtype=float)
        if state.size == 0 or np.any(~np.isfinite(state)):
            raise ValueError("belief state must contain finite values.")
        if covariance.shape != (state.size, state.size) or np.any(~np.isfinite(covariance)):
            raise ValueError("belief covariance must be a finite square matrix matching the state.")
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
            raise ValueError("belief covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(covariance)) < -1.0e-14:
            raise ValueError("belief covariance must be positive semidefinite.")
        if self.process_noise_diag.size != state.size or self.meas_noise_diag.size != state.size:
            raise ValueError("UKF noise vectors must match the belief-state dimension.")
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(measurement.t_s)
            if not np.isfinite(meas_t_s) or meas_t_s < belief_t_s or meas_t_s > output_t_s:
                raise ValueError("measurement epoch must lie within the belief-to-output interval.")
        x_pred, p_pred, sigma_pred, wm, wc = self._predict(
            belief.state,
            belief.covariance,
            from_t_s=float(belief.last_update_t_s),
            to_t_s=meas_t_s,
        )

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred, _, _, _ = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if np.any(~np.isfinite(z)):
            raise ValueError("measurement vector must contain only finite values.")
        if z.size < x_pred.size:
            if meas_t_s < output_t_s:
                x_pred, p_pred, _, _, _ = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        n = x_pred.size
        r = np.diag(self.meas_noise_diag)
        # The standalone orbit measurement is the identity transform. Use the
        # complete predicted covariance so additive process noise participates
        # in both innovation covariance and state/measurement cross covariance.
        z_pred = x_pred
        s_mat = p_pred + r
        pxz = p_pred

        try:
            k_gain = np.linalg.solve(s_mat.T, pxz.T).T
        except np.linalg.LinAlgError:
            k_gain = pxz @ np.linalg.pinv(s_mat)
        innovation = z[:n] - z_pred
        x_upd = x_pred + k_gain @ innovation
        i_kh = np.eye(n) - k_gain
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        if meas_t_s < output_t_s:
            x_upd, p_upd, _, _, _ = self._predict(x_upd, p_upd, from_t_s=meas_t_s, to_t_s=output_t_s)
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=output_t_s)

    def _predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        *,
        from_t_s: float,
        to_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = np.asarray(state, dtype=float).reshape(-1).size
        lam = self.alpha**2 * (n + self.kappa) - n
        if not np.isfinite(lam) or n + lam <= 0.0:
            raise ValueError("UKF alpha/kappa parameters produce invalid sigma-point weights.")
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)

        sigma = self._sigma_points(state, covariance, lam)
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        sigma_pred = None
        if self.numeric_backend == "rust":
            from sim.rust_relative_backend import try_orbit_trial_batch

            sigma_pred = try_orbit_trial_batch(self.propagator, sigma, [dt_s], self.context)
        if sigma_pred is None:
            sigma_pred = np.array(
                [
                    self.propagator.propagate(
                        x_eci=s,
                        dt_s=dt_s,
                        t_s=float(from_t_s),
                        command_accel_eci_km_s2=np.zeros(3),
                        env={},
                        ctx=self.context,
                    )
                    for s in sigma
                ]
            )

        x_pred = np.sum(wm[:, None] * sigma_pred, axis=0)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        process_covariance = np.diag(self.process_noise_diag) * max(q_scale, 0.0)
        if self.numeric_backend == "rust":
            x_pred, p_pred = _load_rust_estimation_backend().ukf_recombine(
                sigma_pred,
                wm,
                wc,
                process_covariance,
            )
        else:
            p_pred = process_covariance
            for i in range(2 * n + 1):
                dx = sigma_pred[i] - x_pred
                p_pred += wc[i] * np.outer(dx, dx)
            p_pred = 0.5 * (p_pred + p_pred.T)
        return x_pred, p_pred, sigma_pred, wm, wc

    def _sigma_points(self, x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
        n = x.size
        if self.numeric_backend == "rust":
            points, _wm, _wc = _load_rust_estimation_backend().ukf_sigma_points(
                x,
                p,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )
            return points
        c = np.linalg.cholesky((n + lam) * p + 1e-12 * np.eye(n))
        points = [x]
        for i in range(n):
            points.append(x + c[:, i])
            points.append(x - c[:, i])
        return np.array(points)
