from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import numpy as np

from sim.acceleration.settings import acceleration_settings_from_mode
from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.two_body import propagate_two_body_rk4
from sim.estimation.innovation_recovery import (
    FullStateInnovationRecovery,
    InnovationRecoveryConfig,
    InnovationRecoveryDiagnostics,
    validate_recovery_config,
)
from sim.numeric_backend import normalize_numeric_backend

orbit_ekf_numerical_jacobian_kernel = None
propagate_two_body_rk4_kernel = None
rust_estimation_backend = None
_ESTIMATION_KERNEL_LOCK = RLock()
_RUST_ESTIMATION_BACKEND_LOCK = RLock()


def _solve_innovation_gain_and_vector(
    innovation_covariance: np.ndarray,
    state_measurement_cross_covariance: np.ndarray,
    innovation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve gain and NIS systems with one factorization when S is exactly symmetric."""

    matrix = np.asarray(innovation_covariance, dtype=float)
    cross = np.asarray(state_measurement_cross_covariance, dtype=float)
    vector = np.asarray(innovation, dtype=float).reshape(-1)
    if np.array_equal(matrix, matrix.T):
        state_dimension = int(cross.shape[0])
        solved = np.linalg.solve(matrix, np.column_stack((cross.T, vector)))
        return solved[:, :state_dimension].T, solved[:, state_dimension]
    return np.linalg.solve(matrix.T, cross.T).T, np.linalg.solve(matrix, vector)


def _load_acceleration_kernels() -> None:
    global orbit_ekf_numerical_jacobian_kernel, propagate_two_body_rk4_kernel
    if propagate_two_body_rk4_kernel is not None:
        return
    with _ESTIMATION_KERNEL_LOCK:
        if propagate_two_body_rk4_kernel is not None:
            return
        from sim.acceleration.kernels.estimation import (
            orbit_ekf_numerical_jacobian_kernel as accelerated_jacobian,
        )
        from sim.acceleration.kernels.estimation import (
            propagate_two_body_rk4_kernel as accelerated_propagate,
        )

        orbit_ekf_numerical_jacobian_kernel = accelerated_jacobian
        propagate_two_body_rk4_kernel = accelerated_propagate


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


@dataclass(frozen=True)
class OrbitEKFUpdateDiagnostics:
    measurement_available: bool
    update_applied: bool
    innovation: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.full((6, 6), np.nan))
    nis: float = float("nan")
    predicted_cov_trace: float = float("nan")
    posterior_cov_trace: float = float("nan")
    recovery_diagnostics: InnovationRecoveryDiagnostics | None = None


@dataclass
class OrbitEKFEstimator(Estimator):
    mu_km3_s2: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    last_update_diagnostics: OrbitEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)
    innovation_recovery: InnovationRecoveryConfig | None = field(default=None, kw_only=True)
    _innovation_recovery: FullStateInnovationRecovery = field(default_factory=FullStateInnovationRecovery, init=False, repr=False)
    acceleration_mode: str = "off"
    numeric_backend: str = "rust"
    _q: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _r: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _i6: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _zero_accel: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False, repr=False)
    _acceleration_enabled_value: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.mu_km3_s2)) or float(self.mu_km3_s2) < 0.0:
            raise ValueError("mu_km3_s2 must be non-negative and finite.")
        if not np.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be positive and finite.")
        self.numeric_backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
        if self.numeric_backend not in {"python", "rust"}:
            raise ValueError("numeric_backend must be 'python' or 'rust'.")
        self.process_noise_diag = np.array(self.process_noise_diag, dtype=float).reshape(-1)
        self.meas_noise_diag = np.array(self.meas_noise_diag, dtype=float).reshape(-1)
        for name, values in (("process_noise_diag", self.process_noise_diag), ("meas_noise_diag", self.meas_noise_diag)):
            if values.size != 6 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain six finite nonnegative values.")
        self._q = np.diag(self.process_noise_diag)
        self._r = np.diag(self.meas_noise_diag)
        self._acceleration_enabled_value = bool(acceleration_settings_from_mode(self.acceleration_mode).enabled)
        if self._acceleration_enabled_value:
            _load_acceleration_kernels()

        validate_recovery_config(self.innovation_recovery, "state")

    def reset_innovation_recovery(self) -> None:
        """Discard pending anomaly history after an external track/reference reset."""
        self._innovation_recovery.reset()

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if self.innovation_recovery is not None and self.innovation_recovery.enabled:
            return self._innovation_recovery.update(self, belief, measurement, t_s)
        self._innovation_recovery.reset()
        return self._update_standard(belief, measurement, t_s)

    def _update_standard(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s = float(t_s)
        belief_t_s = float(belief.last_update_t_s)
        if not np.isfinite(output_t_s) or not np.isfinite(belief_t_s) or output_t_s < belief_t_s:
            raise ValueError("output epoch must be finite and not precede the belief epoch.")
        meas_t_s = output_t_s
        if measurement is not None:
            meas_t_s = float(measurement.t_s)
            if not np.isfinite(meas_t_s) or meas_t_s < belief_t_s or meas_t_s > output_t_s:
                raise ValueError("measurement epoch must lie within the belief-to-output interval.")

        x_pred, p_pred = self._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=meas_t_s)

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=False,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        if z.size < 6:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
                measurement_available=True,
                update_applied=False,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = z[:6]
        if not np.all(np.isfinite(z)):
            raise ValueError("orbit measurement vector must contain finite values.")
        y = z - x_pred
        s = p_pred + self._r
        if self.numeric_backend == "rust":
            # The native path owns the repeated Joseph update and SPD solve.
            # Preserve the Python pseudoinverse behavior only for the
            # singular innovation matrices that the native Cholesky kernel
            # deliberately rejects.
            try:
                x_upd, p_upd, y, s, nis = _load_rust_estimation_backend().ekf_update(
                    x_pred,
                    p_pred,
                    z,
                    self._i6,
                    self._r,
                )
            except ValueError:
                hp_t = p_pred
                try:
                    k, s_y = _solve_innovation_gain_and_vector(s, hp_t, y)
                except np.linalg.LinAlgError:
                    s_pinv = np.linalg.pinv(s)
                    k = hp_t @ s_pinv
                    s_y = s_pinv @ y
                x_upd = x_pred + k @ y
                i_kh = self._i6 - k
                p_upd = i_kh @ p_pred @ i_kh.T + k @ self._r @ k.T
                p_upd = 0.5 * (p_upd + p_upd.T)
                nis = float(y.T @ s_y)
        else:
            hp_t = p_pred
            try:
                k, s_y = _solve_innovation_gain_and_vector(s, hp_t, y)
            except np.linalg.LinAlgError:
                s_pinv = np.linalg.pinv(s)
                k = hp_t @ s_pinv
                s_y = s_pinv @ y
            x_upd = x_pred + k @ y
            i_kh = self._i6 - k
            p_upd = i_kh @ p_pred @ i_kh.T + k @ self._r @ k.T
            p_upd = 0.5 * (p_upd + p_upd.T)
            nis = float(y.T @ s_y)
        self.last_update_diagnostics = OrbitEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            innovation=np.array(y, dtype=float),
            innovation_covariance=np.array(s, dtype=float),
            nis=nis,
            predicted_cov_trace=float(np.trace(p_pred)),
            posterior_cov_trace=float(np.trace(p_upd)),
        )
        if meas_t_s < output_t_s:
            x_upd, p_upd = self._predict(x_upd, p_upd, from_t_s=meas_t_s, to_t_s=output_t_s)
        return StateBelief(state=x_upd, covariance=p_upd, last_update_t_s=output_t_s)

    def _predict(
        self,
        x_prev: np.ndarray,
        p_prev: np.ndarray,
        *,
        from_t_s: float,
        to_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        native = _load_rust_estimation_backend() if self.numeric_backend == "rust" else None
        if native is not None and not self._acceleration_enabled() and np.any(np.asarray(x_prev)[:3] != 0.0):
            # Older optional wheels retain the existing scalar propagation and
            # native Jacobian assembly. The zero-position Python convention
            # also remains with its scalar owner.
            try:
                x_pred, f = native.two_body_predict_and_jacobian(x_prev, dt_s, self.mu_km3_s2)
            except RuntimeError:
                x_pred = self._propagate_state(x_prev, dt_s=dt_s)
                f = self._numerical_jacobian(x_prev, base=x_pred, dt_s=dt_s)
        else:
            x_pred = self._propagate_state(x_prev, dt_s=dt_s)
            f = self._numerical_jacobian(x_prev, base=x_pred, dt_s=dt_s)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = f @ p_prev @ f.T + self._q * max(q_scale, 0.0)
        return x_pred, 0.5 * (p_pred + p_pred.T)

    def _acceleration_enabled(self) -> bool:
        return self._acceleration_enabled_value

    def _propagate_state(self, x: np.ndarray, *, dt_s: float) -> np.ndarray:
        step_dt_s = float(dt_s)
        if self._acceleration_enabled():
            _load_acceleration_kernels()
            return propagate_two_body_rk4_kernel(np.asarray(x, dtype=float).reshape(6), step_dt_s, float(self.mu_km3_s2))
        return propagate_two_body_rk4(
            x_eci=x,
            dt_s=step_dt_s,
            mu_km3_s2=self.mu_km3_s2,
            accel_cmd_eci_km_s2=self._zero_accel,
        )

    def _numerical_jacobian(
        self, x: np.ndarray, *, base: np.ndarray | None = None, dt_s: float | None = None
    ) -> np.ndarray:
        step_dt_s = self.dt_s if dt_s is None else float(dt_s)
        eps = 1e-6
        base_eval = base
        if base_eval is None:
            base_eval = self._propagate_state(x, dt_s=step_dt_s)
        if self.numeric_backend == "rust":
            perturbed = np.empty((6, 6), dtype=float)
            for i in range(6):
                xp = np.asarray(x, dtype=float).reshape(6).copy()
                xp[i] += eps
                perturbed[i] = self._propagate_state(xp, dt_s=step_dt_s)
            j = _load_rust_estimation_backend().jacobian_from_perturbed(
                np.asarray(base_eval, dtype=float).reshape(6),
                perturbed,
                eps,
            )
            return self._i6.copy() if step_dt_s == 0.0 else j
        if self._acceleration_enabled():
            _load_acceleration_kernels()
            j = orbit_ekf_numerical_jacobian_kernel(
                np.asarray(x, dtype=float).reshape(6),
                np.asarray(base_eval, dtype=float).reshape(6),
                step_dt_s,
                float(self.mu_km3_s2),
            )
            # Keep this route's existing nonfinite-result behavior.
            return self._i6.copy() if step_dt_s == 0.0 and np.all(np.isfinite(j)) else j
        j = np.zeros((6, 6))
        for i in range(6):
            xp = x.copy()
            xp[i] += eps
            yp = propagate_two_body_rk4(
                x_eci=xp,
                dt_s=step_dt_s,
                mu_km3_s2=self.mu_km3_s2,
                accel_cmd_eci_km_s2=self._zero_accel,
            )
            j[:, i] = (yp - base_eval) / eps
        # Evaluate the existing route first, preserving its validation and
        # nonfinite-result behavior; valid zero-time flows have exact I6.
        return self._i6.copy() if step_dt_s == 0.0 and np.all(np.isfinite(j)) else j
