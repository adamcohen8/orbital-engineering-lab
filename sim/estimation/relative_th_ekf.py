from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.elements import rv_to_coe_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.estimation.innovation_recovery import (
    FullStateInnovationRecovery,
    InnovationRecoveryConfig,
    InnovationRecoveryDiagnostics,
    validate_recovery_config,
)
from sim.estimation.orbit_ekf import _solve_innovation_gain_and_vector
from sim.estimation.relative_hcw_ekf import (
    _default_measurement_noise_for_model,
    _diag6,
    _measurement_covariance,
    _measurement_innovation,
    _normalize_measurement_origin,
    _validated_update_epochs,
    hcw_measurement_dimension,
    hcw_measurement_jacobian,
    hcw_measurement_vector,
    normalize_hcw_measurement_model,
)
from sim.numeric_backend import normalize_numeric_backend


@dataclass(frozen=True)
class THRelativeEKFUpdateDiagnostics:
    measurement_available: bool
    update_applied: bool
    measurement_model: str = "relative_state"
    innovation: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.full((6, 6), np.nan))
    nis: float = float("nan")
    predicted_cov_trace: float = float("nan")
    posterior_cov_trace: float = float("nan")
    recovery_diagnostics: InnovationRecoveryDiagnostics | None = None


@dataclass
class THRelativeEKFEstimator(Estimator):
    """EKF over rectangular RIC relative state for eccentric-chief RPO.

    The native state is deputy relative to chief in the chief-centered RIC frame:
    [R, I, C, Rdot, Idot, Cdot], using km and km/s. Propagation numerically
    integrates the linearized Tschauner-Hempel equations along a two-body chief
    reference and estimates covariance with the corresponding variational
    state-transition matrix by default. A finite-difference STM remains
    available as an explicit diagnostic/compatibility option. This is the same eccentric-orbit model family
    often used with Yamanaka-Ankersen state-transition solutions.
    """

    chief_state_eci_km_s: np.ndarray
    chief_epoch_t_s: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    measurement_model: str = "relative_state"
    measurement_origin: str = "chief"
    mu_km3_s2: float = EARTH_MU_KM3_S2
    integration_substep_s: float = 10.0
    transition_model: str = "variational_stm"
    meas_noise_covariance: np.ndarray | None = None
    numeric_backend: str = "rust"
    last_update_diagnostics: THRelativeEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)
    innovation_recovery: InnovationRecoveryConfig | None = field(default=None, kw_only=True)
    _innovation_recovery: FullStateInnovationRecovery = field(default_factory=FullStateInnovationRecovery, init=False, repr=False)

    def __post_init__(self) -> None:
        self.chief_state_eci_km_s = _state6(self.chief_state_eci_km_s, "chief_state_eci_km_s")
        self.chief_epoch_t_s = float(self.chief_epoch_t_s)
        if not np.isfinite(self.chief_epoch_t_s):
            raise ValueError("chief_epoch_t_s must be finite.")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if not np.isfinite(self.mu_km3_s2) or self.mu_km3_s2 <= 0.0:
            raise ValueError("mu_km3_s2 must be positive.")
        if not np.isfinite(self.integration_substep_s) or self.integration_substep_s <= 0.0:
            raise ValueError("integration_substep_s must be positive.")
        self.transition_model = _normalize_transition_model(self.transition_model)
        self.numeric_backend = _normalize_numeric_backend(self.numeric_backend)
        self.process_noise_diag = _diag6(self.process_noise_diag, "process_noise_diag")
        self.measurement_model = normalize_hcw_measurement_model(self.measurement_model)
        validate_recovery_config(self.innovation_recovery, self.measurement_model)
        self.measurement_origin = _normalize_measurement_origin(self.measurement_origin)
        meas_dim = hcw_measurement_dimension(self.measurement_model)
        meas_noise = np.array(self.meas_noise_diag, dtype=float).reshape(-1)
        if meas_noise.size == 6 and meas_dim != 6:
            meas_noise = _default_measurement_noise_for_model(self.measurement_model, meas_noise)
        if meas_noise.size != meas_dim:
            raise ValueError(
                f"meas_noise_diag must be length-{meas_dim} for measurement_model={self.measurement_model!r}."
            )
        if np.any(~np.isfinite(meas_noise)) or np.any(meas_noise < 0.0):
            raise ValueError("meas_noise_diag must be finite and non-negative.")
        self.meas_noise_diag = meas_noise
        if self.meas_noise_covariance is not None:
            self.meas_noise_covariance = _measurement_covariance(
                self.meas_noise_covariance, meas_dim
            )

    def set_measurement_covariance(self, covariance: np.ndarray | None) -> None:
        """Set the full covariance used by subsequent measurement updates."""

        self.meas_noise_covariance = (
            None
            if covariance is None
            else _measurement_covariance(
                covariance, hcw_measurement_dimension(self.measurement_model)
            )
        )

    def set_reference_state(self, chief_state_eci_km_s: np.ndarray, t_s: float) -> None:
        """Reset the chief reference used for the next prediction interval."""

        self.chief_state_eci_km_s = _state6(chief_state_eci_km_s, "chief_state_eci_km_s")
        self.chief_epoch_t_s = float(t_s)

    def reset_innovation_recovery(self) -> None:
        """Discard pending anomaly history after an external track/reference reset."""
        self._innovation_recovery.reset()

    def update(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        if self.innovation_recovery is not None and self.innovation_recovery.enabled:
            return self._innovation_recovery.update(self, belief, measurement, t_s)
        self._innovation_recovery.reset()
        return self._update_standard(belief, measurement, t_s)

    def _update_standard(self, belief: StateBelief, measurement: Measurement | None, t_s: float) -> StateBelief:
        output_t_s, meas_t_s = _validated_update_epochs(belief, measurement, t_s)

        x_pred, p_pred = self._predict(belief.state, belief.covariance, from_t_s=belief.last_update_t_s, to_t_s=meas_t_s)

        if measurement is None:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = THRelativeEKFUpdateDiagnostics(
                measurement_available=False,
                update_applied=False,
                measurement_model=self.measurement_model,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)

        z = np.asarray(measurement.vector, dtype=float).reshape(-1)
        h_pred = hcw_measurement_vector(
            self.measurement_model,
            x_pred,
            measurement_origin=self.measurement_origin,
        )
        if z.size < h_pred.size:
            if meas_t_s < output_t_s:
                x_pred, p_pred = self._predict(x_pred, p_pred, from_t_s=meas_t_s, to_t_s=output_t_s)
            self.last_update_diagnostics = THRelativeEKFUpdateDiagnostics(
                measurement_available=True,
                update_applied=False,
                measurement_model=self.measurement_model,
                predicted_cov_trace=float(np.trace(p_pred)),
                posterior_cov_trace=float(np.trace(p_pred)),
            )
            return StateBelief(state=x_pred, covariance=p_pred, last_update_t_s=output_t_s)
        z = z[: h_pred.size]
        if not np.all(np.isfinite(z)):
            raise ValueError("relative measurement vector must contain finite values.")
        h_jac = hcw_measurement_jacobian(
            self.measurement_model,
            x_pred,
            measurement_origin=self.measurement_origin,
        )
        r_mat = (
            np.diag(self.meas_noise_diag)
            if self.meas_noise_covariance is None
            else self.meas_noise_covariance
        )
        innovation = _measurement_innovation(self.measurement_model, z, h_pred)
        s_mat = h_jac @ p_pred @ h_jac.T + r_mat
        hp_t = p_pred @ h_jac.T
        try:
            k_gain, s_y = _solve_innovation_gain_and_vector(s_mat, hp_t, innovation)
        except np.linalg.LinAlgError:
            s_pinv = np.linalg.pinv(s_mat)
            k_gain = hp_t @ s_pinv
            s_y = s_pinv @ innovation
        x_upd = x_pred + k_gain @ innovation
        i_kh = np.eye(6) - k_gain @ h_jac
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r_mat @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        self.last_update_diagnostics = THRelativeEKFUpdateDiagnostics(
            measurement_available=True,
            update_applied=True,
            measurement_model=self.measurement_model,
            innovation=np.array(innovation, dtype=float),
            innovation_covariance=np.array(s_mat, dtype=float),
            nis=float(innovation.T @ s_y),
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
        from_t = float(from_t_s)
        to_t = float(to_t_s)
        dt_s = max(to_t - from_t, 0.0)
        x = np.asarray(x_prev, dtype=float).reshape(6)
        p = np.asarray(p_prev, dtype=float).reshape(6, 6)
        if dt_s <= 0.0:
            return x.copy(), 0.5 * (p + p.T)

        chief_start = self._chief_state_at(from_t)
        if self.numeric_backend == "rust":
            from sim.rust_relative_backend import (
                th_finite_difference_transition_matrix as rust_th_finite_difference_transition_matrix,
            )
            from sim.rust_relative_backend import (
                th_propagate_relative_state as rust_th_propagate_relative_state,
            )
            from sim.rust_relative_backend import (
                th_variational_propagate_relative_state_and_stm as rust_th_variational_propagate_relative_state_and_stm,
            )
            from sim.rust_relative_backend import (
                ya_propagate_relative_state_and_stm as rust_ya_propagate_relative_state_and_stm,
            )

            if self.transition_model == "closed_form_ya":
                x_pred, phi = rust_ya_propagate_relative_state_and_stm(
                    x,
                    dt_s,
                    chief_start,
                    mu_km3_s2=float(self.mu_km3_s2),
                    max_step_s=float(self.integration_substep_s),
                )
            elif self.transition_model == "variational_stm":
                x_pred, phi = rust_th_variational_propagate_relative_state_and_stm(
                    x,
                    dt_s,
                    chief_start,
                    mu_km3_s2=float(self.mu_km3_s2),
                    max_step_s=float(self.integration_substep_s),
                )
            else:
                x_pred = rust_th_propagate_relative_state(
                    x,
                    dt_s,
                    chief_start,
                    mu_km3_s2=float(self.mu_km3_s2),
                    max_step_s=float(self.integration_substep_s),
                )
                phi = rust_th_finite_difference_transition_matrix(
                    x,
                    dt_s,
                    chief_start,
                    mu_km3_s2=float(self.mu_km3_s2),
                    max_step_s=float(self.integration_substep_s),
                )
        elif self.transition_model == "closed_form_ya":
            x_pred, phi = ya_closed_form_propagate_relative_state_and_stm(
                x,
                dt_s,
                chief_start,
                mu_km3_s2=float(self.mu_km3_s2),
                max_step_s=float(self.integration_substep_s),
                numeric_backend=self.numeric_backend,
            )
        elif self.transition_model == "variational_stm":
            x_pred, phi = th_variational_propagate_relative_state_and_stm(
                x,
                dt_s,
                chief_start,
                mu_km3_s2=float(self.mu_km3_s2),
                max_step_s=float(self.integration_substep_s),
                numeric_backend=self.numeric_backend,
            )
        else:
            x_pred = th_propagate_relative_state(
                x,
                dt_s,
                chief_start,
                mu_km3_s2=float(self.mu_km3_s2),
                max_step_s=float(self.integration_substep_s),
                numeric_backend=self.numeric_backend,
            )
            phi = th_relative_transition_matrix(
                x,
                dt_s,
                chief_start,
                mu_km3_s2=float(self.mu_km3_s2),
                max_step_s=float(self.integration_substep_s),
                numeric_backend=self.numeric_backend,
            )
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = phi @ p @ phi.T + np.diag(self.process_noise_diag) * max(q_scale, 0.0)
        return x_pred, 0.5 * (p_pred + p_pred.T)

    def _chief_state_at(self, t_s: float) -> np.ndarray:
        dt_s = float(t_s) - float(self.chief_epoch_t_s)
        if self.numeric_backend == "rust" and np.isfinite(dt_s):
            from sim.rust_relative_backend import try_chief_state_propagation

            native = try_chief_state_propagation(
                self.chief_state_eci_km_s,
                dt_s,
                mu_km3_s2=float(self.mu_km3_s2),
                max_step_s=float(self.integration_substep_s),
            )
            if native is not None:
                return native
        return _propagate_chief_state(
            self.chief_state_eci_km_s,
            dt_s,
            mu_km3_s2=float(self.mu_km3_s2),
            max_step_s=float(self.integration_substep_s),
        )


@dataclass
class YARelativeEKFEstimator(THRelativeEKFEstimator):
    """Relative EKF using the closed-form Yamanaka-Ankersen STM.

    This estimator uses the same linearized Tschauner-Hempel relative dynamics
    as `THRelativeEKFEstimator`, but propagates state and covariance through
    the compact anomaly-domain Yamanaka-Ankersen STM from the TH solution. OEL's
    runtime state remains rectangular RIC [R, I, C, Rdot, Idot, Cdot], so the YA
    STM is sandwiched between deterministic conversions to and from normalized
    true-anomaly variables.
    """

    transition_model: str = "closed_form_ya"


def th_propagate_relative_state(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_step_s: float = 10.0,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Propagate linearized eccentric-chief RIC relative motion."""

    rel = _state6(relative_state_ric, "relative_state_ric")
    chief = _state6(chief_state_eci_km_s, "chief_state_eci_km_s")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import th_propagate_relative_state as rust_th_propagate

        return rust_th_propagate(
            rel,
            float(dt_s),
            chief,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(max_step_s),
        )
    state = np.hstack((chief, rel)).astype(float)
    propagated = _integrate_state(state, float(dt_s), float(mu_km3_s2), float(max_step_s), _th_combined_derivative)
    return propagated[6:12]


def th_relative_transition_matrix(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_step_s: float = 10.0,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Return a finite-difference STM for the TH-integrated relative propagator."""

    x = _state6(relative_state_ric, "relative_state_ric")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import th_finite_difference_transition_matrix

        return th_finite_difference_transition_matrix(
            x,
            float(dt_s),
            _state6(chief_state_eci_km_s, "chief_state_eci_km_s"),
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(max_step_s),
        )
    phi = np.zeros((6, 6), dtype=float)
    base = th_propagate_relative_state(
        x,
        dt_s,
        chief_state_eci_km_s,
        mu_km3_s2=mu_km3_s2,
        max_step_s=max_step_s,
        numeric_backend=numeric_backend,
    )
    eps = np.array([1e-5, 1e-5, 1e-5, 1e-8, 1e-8, 1e-8], dtype=float)
    for idx in range(6):
        xp = x.copy()
        xp[idx] += eps[idx]
        hp = th_propagate_relative_state(
            xp,
            dt_s,
            chief_state_eci_km_s,
            mu_km3_s2=mu_km3_s2,
            max_step_s=max_step_s,
            numeric_backend=numeric_backend,
        )
        phi[:, idx] = (hp - base) / eps[idx]
    return phi


def th_variational_propagate_relative_state_and_stm(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_step_s: float = 10.0,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate relative state and STM through the TH variational equations."""

    rel = _state6(relative_state_ric, "relative_state_ric")
    chief = _state6(chief_state_eci_km_s, "chief_state_eci_km_s")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import th_variational_propagate_relative_state_and_stm as rust_th_variational

        return rust_th_variational(
            rel,
            float(dt_s),
            chief,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(max_step_s),
        )
    phi0 = np.eye(6, dtype=float).reshape(36)
    state = np.hstack((chief, rel, phi0)).astype(float)
    propagated = _integrate_state(
        state,
        float(dt_s),
        float(mu_km3_s2),
        float(max_step_s),
        _th_variational_combined_derivative,
    )
    return propagated[6:12], propagated[12:48].reshape(6, 6)


def th_variational_transition_matrix(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_step_s: float = 10.0,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Return the variational STM for linearized eccentric-chief RIC motion."""

    # Preserve validation of the public argument even though the linear TH STM
    # is independent of the relative state.  Integrating only chief+STM avoids
    # carrying six identically-zero relative-state entries through every RK4
    # stage while retaining the exact chief and STM operation ordering.
    _state6(relative_state_ric, "relative_state_ric")
    chief = _state6(chief_state_eci_km_s, "chief_state_eci_km_s")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import th_variational_propagate_relative_state_and_stm as rust_th_variational

        return rust_th_variational(
            np.zeros(6, dtype=float),
            float(dt_s),
            chief,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(max_step_s),
        )[1]
    state = np.hstack((chief, np.eye(6, dtype=float).reshape(36))).astype(float)
    propagated = _integrate_state(
        state,
        float(dt_s),
        float(mu_km3_s2),
        float(max_step_s),
        _th_variational_stm_derivative,
    )
    return propagated[6:42].reshape(6, 6)


def ya_closed_form_propagate_relative_state_and_stm(
    relative_state_ric: np.ndarray,
    dt_s: float,
    chief_state_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    max_step_s: float = 10.0,
    numeric_backend: str = "rust",
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate RIC relative state with the closed-form YA STM.

    The Yamanaka-Ankersen matrix acts on normalized variables ordered as
    [xbar, xbar', ybar, ybar', zbar, zbar'], where bars divide by chief radius
    and primes denote derivatives with respect to true anomaly. OEL exposes
    rectangular RIC states in km and km/s, so this helper returns the equivalent
    dimensional state transition matrix for [R, I, C, Rdot, Idot, Cdot].
    """

    rel = _state6(relative_state_ric, "relative_state_ric")
    chief0 = _state6(chief_state_eci_km_s, "chief_state_eci_km_s")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import ya_propagate_relative_state_and_stm as rust_ya_propagate

        return rust_ya_propagate(
            rel,
            float(dt_s),
            chief0,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=float(max_step_s),
        )
    chief1 = _propagate_chief_state(
        chief0,
        float(dt_s),
        mu_km3_s2=float(mu_km3_s2),
        max_step_s=float(max_step_s),
    )
    phi = ya_closed_form_transition_matrix(
        float(dt_s),
        chief0,
        chief1,
        mu_km3_s2=float(mu_km3_s2),
        numeric_backend=numeric_backend,
    )
    return phi @ rel, phi


def ya_closed_form_transition_matrix(
    dt_s: float,
    chief_start_eci_km_s: np.ndarray,
    chief_end_eci_km_s: np.ndarray | None = None,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    numeric_backend: str = "rust",
) -> np.ndarray:
    """Return the dimensional RIC STM implied by the Yamanaka-Ankersen solution."""

    chief0 = _state6(chief_start_eci_km_s, "chief_start_eci_km_s")
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import ya_state_transition_matrix

        chief1 = None if chief_end_eci_km_s is None else _state6(chief_end_eci_km_s, "chief_end_eci_km_s")
        return ya_state_transition_matrix(
            float(dt_s),
            chief0,
            chief1,
            mu_km3_s2=float(mu_km3_s2),
            max_step_s=10.0,
        )
    chief1 = (
        _propagate_chief_state(chief0, float(dt_s), mu_km3_s2=float(mu_km3_s2), max_step_s=10.0)
        if chief_end_eci_km_s is None
        else _state6(chief_end_eci_km_s, "chief_end_eci_km_s")
    )
    normalized_phi = ya_normalized_state_transition_matrix(
        float(dt_s),
        chief0,
        chief1,
        mu_km3_s2=float(mu_km3_s2),
    )
    to_normalized = _ric_to_ya_normalized_matrix(chief0)
    from_normalized = _ya_normalized_to_ric_matrix(chief1)
    return from_normalized @ normalized_phi @ to_normalized


def ya_normalized_state_transition_matrix(
    dt_s: float,
    chief_start_eci_km_s: np.ndarray,
    chief_end_eci_km_s: np.ndarray,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> np.ndarray:
    """Return YA STM in [xbar, xbar', ybar, ybar', zbar, zbar'] ordering."""

    chief0 = _state6(chief_start_eci_km_s, "chief_start_eci_km_s")
    chief1 = _state6(chief_end_eci_km_s, "chief_end_eci_km_s")
    e0, f0 = _chief_eccentricity_and_true_anomaly(chief0, float(mu_km3_s2))
    e1, f1 = _chief_eccentricity_and_true_anomaly(chief1, float(mu_km3_s2))
    if abs(e1 - e0) > 1.0e-7:
        raise ValueError("YA STM requires a two-body chief with constant eccentricity.")
    h_norm = float(np.linalg.norm(np.cross(chief0[:3], chief0[3:6])))
    if h_norm <= 1.0e-12:
        raise ValueError("YA STM requires nonzero chief angular momentum.")
    integral_i = (float(mu_km3_s2) ** 2) * float(dt_s) / (h_norm**3)
    return _ya_phi(float(e0), float(f1), integral_i) @ _ya_phi_inverse_at_initial(float(e0), float(f0))


def _propagate_chief_state(
    chief_state_eci_km_s: np.ndarray,
    dt_s: float,
    *,
    mu_km3_s2: float,
    max_step_s: float,
) -> np.ndarray:
    return _integrate_state(
        _state6(chief_state_eci_km_s, "chief_state_eci_km_s"),
        float(dt_s),
        float(mu_km3_s2),
        float(max_step_s),
        _two_body_derivative,
    )


def _integrate_state(
    state: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    max_step_s: float,
    derivative_func: object,
) -> np.ndarray:
    duration = abs(float(dt_s))
    if duration <= 0.0:
        return np.array(state, dtype=float).copy()
    direction = 1.0 if float(dt_s) >= 0.0 else -1.0
    step_limit = max(float(max_step_s), 1.0e-9)
    elapsed = 0.0
    out = np.array(state, dtype=float).copy()
    while elapsed < duration - 1.0e-12:
        h = min(step_limit, duration - elapsed) * direction
        out = _rk4_step(derivative_func, out, h, mu_km3_s2)
        elapsed += abs(h)
    return out


def _rk4_step(derivative_func: object, state: np.ndarray, step_s: float, mu_km3_s2: float) -> np.ndarray:
    h = float(step_s)
    k1 = derivative_func(state, mu_km3_s2)
    k2 = derivative_func(state + 0.5 * h * k1, mu_km3_s2)
    k3 = derivative_func(state + 0.5 * h * k2, mu_km3_s2)
    k4 = derivative_func(state + h * k3, mu_km3_s2)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _two_body_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    r_vec = np.array(state[:3], dtype=float)
    v_vec = np.array(state[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    return np.hstack((v_vec, -float(mu_km3_s2) * r_vec / (r_norm**3)))


def _th_combined_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    rho = np.array(state[6:9], dtype=float)
    rho_dot = np.array(state[9:12], dtype=float)
    r_norm = float(np.linalg.norm(chief_r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    h_vec = np.cross(chief_r, chief_v)
    h_norm = float(np.linalg.norm(h_vec))
    theta_dot = h_norm / max(r_norm * r_norm, 1.0e-12)
    radial_rate = float(np.dot(chief_r, chief_v)) / r_norm
    theta_ddot = -2.0 * theta_dot * radial_rate / r_norm
    gravity_gradient = (float(mu_km3_s2) / (r_norm**3)) * np.array([2.0 * rho[0], -rho[1], -rho[2]])
    # These are the closed-form cross products for z-axis frame rotation.  Keep
    # the same operation grouping as the vector form so the optimized path is
    # bit-for-bit identical while avoiding four tiny ``np.cross`` dispatches at
    # every RK4 derivative evaluation.
    coriolis_cross = np.array(
        [-theta_dot * rho_dot[1], theta_dot * rho_dot[0], 0.0],
        dtype=float,
    )
    euler_cross = np.array(
        [-theta_ddot * rho[1], theta_ddot * rho[0], 0.0],
        dtype=float,
    )
    omega_cross_rho = np.array(
        [-theta_dot * rho[1], theta_dot * rho[0], 0.0],
        dtype=float,
    )
    centrifugal_cross = np.array(
        [-theta_dot * omega_cross_rho[1], theta_dot * omega_cross_rho[0], 0.0],
        dtype=float,
    )
    rho_ddot = (
        gravity_gradient
        - 2.0 * coriolis_cross
        - euler_cross
        - centrifugal_cross
    )
    chief_acc = -float(mu_km3_s2) * chief_r / (r_norm**3)
    return np.hstack((chief_v, chief_acc, rho_dot, rho_ddot))


def _th_variational_combined_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    base = _th_combined_derivative(state[:12], mu_km3_s2)
    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    phi = np.array(state[12:48], dtype=float).reshape(6, 6)
    a_mat = _th_relative_dynamics_matrix(chief_r, chief_v, float(mu_km3_s2))
    return np.hstack((base, (a_mat @ phi).reshape(36)))


def _th_variational_stm_derivative(state: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Chief+STM derivative used when no relative-state propagation is needed."""

    chief_r = np.array(state[:3], dtype=float)
    chief_v = np.array(state[3:6], dtype=float)
    r_norm = float(np.linalg.norm(chief_r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros_like(state)
    chief_acc = -float(mu_km3_s2) * chief_r / (r_norm**3)
    phi = np.array(state[6:42], dtype=float).reshape(6, 6)
    a_mat = _th_relative_dynamics_matrix(chief_r, chief_v, float(mu_km3_s2))
    return np.hstack((chief_v, chief_acc, (a_mat @ phi).reshape(36)))


def _th_relative_dynamics_matrix(
    chief_r_eci_km: np.ndarray,
    chief_v_eci_km_s: np.ndarray,
    mu_km3_s2: float,
) -> np.ndarray:
    chief_r = np.array(chief_r_eci_km, dtype=float).reshape(3)
    chief_v = np.array(chief_v_eci_km_s, dtype=float).reshape(3)
    r_norm = float(np.linalg.norm(chief_r))
    if r_norm <= 1.0e-9 or not np.isfinite(r_norm):
        return np.zeros((6, 6), dtype=float)
    h_norm = float(np.linalg.norm(np.cross(chief_r, chief_v)))
    theta_dot = h_norm / max(r_norm * r_norm, 1.0e-12)
    radial_rate = float(np.dot(chief_r, chief_v)) / r_norm
    theta_ddot = -2.0 * theta_dot * radial_rate / r_norm
    gravity = float(mu_km3_s2) / (r_norm**3)
    a_mat = np.zeros((6, 6), dtype=float)
    a_mat[:3, 3:] = np.eye(3)
    a_mat[3, 0] = 2.0 * gravity + theta_dot * theta_dot
    a_mat[3, 1] = theta_ddot
    a_mat[3, 4] = 2.0 * theta_dot
    a_mat[4, 0] = -theta_ddot
    a_mat[4, 1] = -gravity + theta_dot * theta_dot
    a_mat[4, 3] = -2.0 * theta_dot
    a_mat[5, 2] = -gravity
    return a_mat


def _chief_eccentricity_and_true_anomaly(chief_state_eci_km_s: np.ndarray, mu_km3_s2: float) -> tuple[float, float]:
    coes = rv_to_coe_eci(chief_state_eci_km_s[:3], chief_state_eci_km_s[3:6], mu_km3_s2=float(mu_km3_s2))
    return float(coes.ecc), float(np.deg2rad(coes.true_anomaly_deg))


def _ric_to_ya_normalized_matrix(chief_state_eci_km_s: np.ndarray) -> np.ndarray:
    r_vec = np.array(chief_state_eci_km_s[:3], dtype=float)
    v_vec = np.array(chief_state_eci_km_s[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r_vec))
    h_norm = float(np.linalg.norm(np.cross(r_vec, v_vec)))
    if r_norm <= 1.0e-12 or h_norm <= 1.0e-12:
        raise ValueError("YA normalized conversion requires nonzero chief radius and angular momentum.")
    radial_rate = float(np.dot(r_vec, v_vec)) / r_norm
    true_anomaly_rate = h_norm / (r_norm * r_norm)
    scale = 1.0 / (r_norm * true_anomaly_rate)

    mat = np.zeros((6, 6), dtype=float)
    for coord_idx, row_idx in ((0, 0), (1, 2), (2, 4)):
        vel_idx = coord_idx + 3
        mat[row_idx, coord_idx] = 1.0 / r_norm
        mat[row_idx + 1, coord_idx] = -radial_rate / (r_norm * r_norm * true_anomaly_rate)
        mat[row_idx + 1, vel_idx] = scale
    return mat


def _ya_normalized_to_ric_matrix(chief_state_eci_km_s: np.ndarray) -> np.ndarray:
    r_vec = np.array(chief_state_eci_km_s[:3], dtype=float)
    v_vec = np.array(chief_state_eci_km_s[3:6], dtype=float)
    r_norm = float(np.linalg.norm(r_vec))
    h_norm = float(np.linalg.norm(np.cross(r_vec, v_vec)))
    if r_norm <= 1.0e-12 or h_norm <= 1.0e-12:
        raise ValueError("YA dimensional conversion requires nonzero chief radius and angular momentum.")
    radial_rate = float(np.dot(r_vec, v_vec)) / r_norm
    true_anomaly_rate = h_norm / (r_norm * r_norm)

    mat = np.zeros((6, 6), dtype=float)
    for coord_idx, row_idx in ((0, 0), (1, 2), (2, 4)):
        vel_idx = coord_idx + 3
        mat[coord_idx, row_idx] = r_norm
        mat[vel_idx, row_idx] = radial_rate
        mat[vel_idx, row_idx + 1] = r_norm * true_anomaly_rate
    return mat


def _ya_phi(ecc: float, true_anomaly_rad: float, integral_i: float) -> np.ndarray:
    e = float(ecc)
    f = float(true_anomaly_rad)
    i_val = float(integral_i)
    sin_f = float(np.sin(f))
    cos_f = float(np.cos(f))
    k = 1.0 + e * cos_f
    if k <= 1.0e-12:
        raise ValueError("YA STM is singular for this chief true anomaly/eccentricity.")
    s = k * sin_f
    c = k * cos_f
    s_prime = k * cos_f - e * sin_f * sin_f
    c_prime = -k * sin_f - e * sin_f * cos_f

    phi = np.zeros((6, 6), dtype=float)
    phi[0, 0] = s
    phi[0, 1] = c
    phi[0, 2] = 2.0 - 3.0 * e * s * i_val
    phi[1, 0] = s_prime
    phi[1, 1] = c_prime
    phi[1, 2] = -3.0 * e * (s_prime * i_val + s / (k * k))
    phi[2, 0] = c * (1.0 + 1.0 / k)
    phi[2, 1] = -s * (1.0 + 1.0 / k)
    phi[2, 2] = -3.0 * k * k * i_val
    phi[2, 3] = 1.0
    phi[3, 0] = -2.0 * s
    phi[3, 1] = e - 2.0 * c
    phi[3, 2] = -3.0 * (1.0 - 2.0 * e * s * i_val)
    phi[4, 4] = cos_f
    phi[4, 5] = sin_f
    phi[5, 4] = -sin_f
    phi[5, 5] = cos_f
    return phi


def _ya_phi_inverse_at_initial(ecc: float, true_anomaly_rad: float) -> np.ndarray:
    e = float(ecc)
    f = float(true_anomaly_rad)
    sin_f = float(np.sin(f))
    cos_f = float(np.cos(f))
    eta_sq = 1.0 - e * e
    if eta_sq <= 1.0e-12:
        raise ValueError("YA STM requires elliptical chief eccentricity below one.")
    k = 1.0 + e * cos_f
    if k <= 1.0e-12:
        raise ValueError("YA STM is singular for this initial true anomaly/eccentricity.")
    s = k * sin_f
    c = k * cos_f

    inv = np.zeros((6, 6), dtype=float)
    inv[0, 0] = -3.0 * s * (k + e * e) / (k * k)
    inv[0, 1] = c - 2.0 * e
    inv[0, 3] = -s * (k + 1.0) / k
    inv[1, 0] = -3.0 * (e + c / k)
    inv[1, 1] = -s
    inv[1, 3] = -(c * (k + 1.0) / k + e)
    inv[2, 0] = 3.0 * k - eta_sq
    inv[2, 1] = e * s
    inv[2, 3] = k * k
    inv[3, 0] = -3.0 * e * s * (k + 1.0) / (k * k)
    inv[3, 1] = -2.0 + e * c
    inv[3, 2] = eta_sq
    inv[3, 3] = -e * s * (k + 1.0) / k
    inv[4, 4] = eta_sq * cos_f
    inv[4, 5] = -eta_sq * sin_f
    inv[5, 4] = eta_sq * sin_f
    inv[5, 5] = eta_sq * cos_f
    return inv / eta_sq


def _normalize_transition_model(value: str) -> str:
    raw = str(value or "finite_difference").strip().lower().replace("-", "_")
    aliases = {
        "fd": "finite_difference",
        "finite_difference_stm": "finite_difference",
        "th": "finite_difference",
        "th_integrated": "finite_difference",
        "ya": "closed_form_ya",
        "ya_stm": "closed_form_ya",
        "yamanaka_ankersen": "closed_form_ya",
        "closed_form": "closed_form_ya",
        "closed_form_ya_stm": "closed_form_ya",
        "variational": "variational_stm",
        "variational_ya": "variational_stm",
        "stm": "variational_stm",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"finite_difference", "variational_stm", "closed_form_ya"}:
        raise ValueError("transition_model must be 'finite_difference', 'variational_stm', or 'closed_form_ya'.")
    return normalized


def _normalize_numeric_backend(value: str) -> str:
    normalized = normalize_numeric_backend("python" if value is None else value, error_message="numeric_backend must be 'python' or 'rust'.")
    return normalized


def _state6(value: np.ndarray, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"{field_name} must be length-6.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} must contain finite values.")
    return arr.astype(float)
