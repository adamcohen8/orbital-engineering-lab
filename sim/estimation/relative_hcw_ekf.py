from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.core.interfaces import Estimator
from sim.core.models import Measurement, StateBelief
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_RADIUS_KM
from sim.dynamics.orbit.relative_linear import RelativeLinearDynamics
from sim.estimation.innovation_recovery import (
    FullStateInnovationRecovery,
    InnovationRecoveryConfig,
    InnovationRecoveryDiagnostics,
    validate_recovery_config,
)
from sim.estimation.orbit_ekf import _solve_innovation_gain_and_vector
from sim.numeric_backend import normalize_numeric_backend

HCW_MEASUREMENT_MODELS = {
    "relative_state",
    "relative_range",
    "relative_range_rate",
    "relative_angles",
    "relative_angles_range",
    "relative_angles_range_rate",
}


@dataclass(frozen=True)
class HCWRelativeEKFUpdateDiagnostics:
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
class HCWRelativeEKFEstimator(Estimator):
    """EKF over rectangular RIC relative state using analytic HCW propagation.

    The native state is deputy relative to chief in the chief-centered RIC frame:
    [R, I, C, Rdot, Idot, Cdot], using km and km/s.
    """

    mean_motion_rad_s: float
    dt_s: float
    process_noise_diag: np.ndarray
    meas_noise_diag: np.ndarray
    measurement_model: str = "relative_state"
    measurement_origin: str = "chief"
    meas_noise_covariance: np.ndarray | None = None
    numeric_backend: str = "rust"
    last_update_diagnostics: HCWRelativeEKFUpdateDiagnostics | None = field(default=None, init=False, repr=False)
    innovation_recovery: InnovationRecoveryConfig | None = field(default=None, kw_only=True)
    _innovation_recovery: FullStateInnovationRecovery = field(default_factory=FullStateInnovationRecovery, init=False, repr=False)
    _q: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _r: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)), init=False, repr=False)
    _i6: np.ndarray = field(default_factory=lambda: np.eye(6), init=False, repr=False)
    _cached_transition_key: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _cached_transition: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean_motion_rad_s) or self.mean_motion_rad_s <= 0.0:
            raise ValueError("mean_motion_rad_s must be positive.")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        self.process_noise_diag = _diag6(self.process_noise_diag, "process_noise_diag")
        self.measurement_model = normalize_hcw_measurement_model(self.measurement_model)
        validate_recovery_config(self.innovation_recovery, self.measurement_model)
        self.measurement_origin = _normalize_measurement_origin(self.measurement_origin)
        self.numeric_backend = _normalize_numeric_backend(self.numeric_backend)
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
        self._q = np.diag(self.process_noise_diag)
        self._r = np.diag(self.meas_noise_diag)
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
            self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
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
            self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
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
        r = (
            self._r
            if self.meas_noise_covariance is None
            else self.meas_noise_covariance
        )
        innovation = _measurement_innovation(self.measurement_model, z, h_pred)
        s_mat = h_jac @ p_pred @ h_jac.T + r
        hp_t = p_pred @ h_jac.T
        try:
            k_gain, s_y = _solve_innovation_gain_and_vector(s_mat, hp_t, innovation)
        except np.linalg.LinAlgError:
            s_pinv = np.linalg.pinv(s_mat)
            k_gain = hp_t @ s_pinv
            s_y = s_pinv @ innovation
        x_upd = x_pred + k_gain @ innovation
        i_kh = self._i6 - k_gain @ h_jac
        p_upd = i_kh @ p_pred @ i_kh.T + k_gain @ r @ k_gain.T
        p_upd = 0.5 * (p_upd + p_upd.T)
        self.last_update_diagnostics = HCWRelativeEKFUpdateDiagnostics(
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
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        transition_key = (float(self.mean_motion_rad_s), dt_s)
        phi = self._cached_transition if self._cached_transition_key == transition_key else None
        if phi is None:
            if self.numeric_backend == "rust":
                from sim.rust_relative_backend import hcw_state_transition_matrix as rust_hcw_state_transition_matrix

                phi = rust_hcw_state_transition_matrix(float(self.mean_motion_rad_s), dt_s)
            else:
                phi = hcw_state_transition_matrix(float(self.mean_motion_rad_s), dt_s, numeric_backend=self.numeric_backend)
            self._cached_transition_key = transition_key
            self._cached_transition = phi
        x = np.asarray(x_prev, dtype=float).reshape(6)
        p = np.asarray(p_prev, dtype=float).reshape(6, 6)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = phi @ p @ phi.T + self._q * max(q_scale, 0.0)
        return phi @ x, 0.5 * (p_pred + p_pred.T)


@dataclass
class SSJ2RelativeEKFEstimator(HCWRelativeEKFEstimator):
    """EKF using homogeneous chief-centered Schweighart-Sedwick dynamics.

    The measurement contract is identical to :class:`HCWRelativeEKFEstimator`.
    The reference orbit must be circular or near-circular and its radius and
    inclination use mean-orbit semantics.
    """

    reference_radius_km: float = 7000.0
    reference_inclination_rad: float = 0.0
    j2: float = EARTH_J2
    earth_radius_km: float = EARTH_RADIUS_KM
    reference_eccentricity: float | None = None
    maximum_supported_eccentricity: float = 0.01
    _relative_dynamics: RelativeLinearDynamics = field(init=False, repr=False)
    def __post_init__(self) -> None:
        super().__post_init__()
        self._relative_dynamics = RelativeLinearDynamics(
            model="ss_j2",
            mean_motion_rad_s=float(self.mean_motion_rad_s),
            reference_radius_km=float(self.reference_radius_km),
            reference_inclination_rad=float(self.reference_inclination_rad),
            j2=float(self.j2),
            earth_radius_km=float(self.earth_radius_km),
            reference_eccentricity=self.reference_eccentricity,
            maximum_supported_eccentricity=float(self.maximum_supported_eccentricity),
            numeric_backend=self.numeric_backend,
        )

    @classmethod
    def from_chief_state(
        cls,
        chief_state_eci_km_s: np.ndarray,
        *,
        dt_s: float,
        process_noise_diag: np.ndarray,
        meas_noise_diag: np.ndarray,
        measurement_model: str = "relative_state",
        measurement_origin: str = "chief",
        j2: float = EARTH_J2,
        earth_radius_km: float = EARTH_RADIUS_KM,
        maximum_supported_eccentricity: float = 0.01,
        numeric_backend: str = "rust",
        innovation_recovery: InnovationRecoveryConfig | None = None,
    ) -> SSJ2RelativeEKFEstimator:
        dynamics = RelativeLinearDynamics.ss_j2_from_chief_state(
            chief_state_eci_km_s,
            j2=j2,
            earth_radius_km=earth_radius_km,
            maximum_supported_eccentricity=maximum_supported_eccentricity,
            numeric_backend=numeric_backend,
        )
        return cls(
            mean_motion_rad_s=dynamics.mean_motion_rad_s,
            dt_s=dt_s,
            process_noise_diag=process_noise_diag,
            meas_noise_diag=meas_noise_diag,
            measurement_model=measurement_model,
            measurement_origin=measurement_origin,
            reference_radius_km=float(dynamics.reference_radius_km),
            reference_inclination_rad=float(dynamics.reference_inclination_rad),
            j2=j2,
            earth_radius_km=earth_radius_km,
            reference_eccentricity=dynamics.reference_eccentricity,
            maximum_supported_eccentricity=maximum_supported_eccentricity,
            numeric_backend=numeric_backend,
            innovation_recovery=innovation_recovery,
        )

    def _predict(
        self,
        x_prev: np.ndarray,
        p_prev: np.ndarray,
        *,
        from_t_s: float,
        to_t_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt_s = max(float(to_t_s) - float(from_t_s), 0.0)
        transition_key = (float(self.mean_motion_rad_s), dt_s)
        phi = self._cached_transition if self._cached_transition_key == transition_key else None
        if phi is None:
            phi = self._relative_dynamics.state_transition_matrix(dt_s)
            self._cached_transition_key = transition_key
            self._cached_transition = phi
        x = np.asarray(x_prev, dtype=float).reshape(6)
        p = np.asarray(p_prev, dtype=float).reshape(6, 6)
        q_scale = dt_s / self.dt_s if self.dt_s > 0.0 else 1.0
        p_pred = phi @ p @ phi.T + self._q * max(q_scale, 0.0)
        return phi @ x, 0.5 * (p_pred + p_pred.T)

    def model_metadata(self) -> dict[str, object]:
        return self._relative_dynamics.metadata()


def hcw_state_transition_matrix(
    mean_motion_rad_s: float,
    dt_s: float,
    *,
    numeric_backend: str = "rust",
) -> np.ndarray:
    if _normalize_numeric_backend(numeric_backend) == "rust":
        from sim.rust_relative_backend import hcw_state_transition_matrix as rust_hcw_state_transition_matrix

        return rust_hcw_state_transition_matrix(float(mean_motion_rad_s), float(dt_s))
    return RelativeLinearDynamics.hcw(float(mean_motion_rad_s), numeric_backend="python").state_transition_matrix(float(dt_s))


def normalize_hcw_measurement_model(model: str) -> str:
    raw = str(model or "relative_state").strip().lower().replace("-", "_")
    aliases = {
        "state": "relative_state",
        "full_state": "relative_state",
        "ric_state": "relative_state",
        "range": "relative_range",
        "range_rate": "relative_range_rate",
        "angles": "relative_angles",
        "angles_range": "relative_angles_range",
        "angles_range_rate": "relative_angles_range_rate",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in HCW_MEASUREMENT_MODELS:
        valid = ", ".join(sorted(HCW_MEASUREMENT_MODELS))
        raise ValueError(f"Unsupported HCW measurement_model '{model}'. Valid options: {valid}")
    return normalized


def hcw_measurement_dimension(model: str) -> int:
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_state":
        return 6
    if normalized == "relative_range":
        return 1
    if normalized in {"relative_range_rate", "relative_angles"}:
        return 2
    if normalized == "relative_angles_range":
        return 3
    if normalized == "relative_angles_range_rate":
        return 4
    raise ValueError(f"Unsupported HCW measurement_model '{model}'.")


def hcw_measurement_vector(model: str, state: np.ndarray, *, measurement_origin: str = "chief") -> np.ndarray:
    normalized = normalize_hcw_measurement_model(model)
    origin = _normalize_measurement_origin(measurement_origin)
    x = np.asarray(state, dtype=float).reshape(6)
    sign = 1.0 if origin == "chief" else -1.0
    if normalized == "relative_state":
        return sign * x
    rel_r = sign * x[:3]
    rel_v = sign * x[3:]
    rng_km = float(np.linalg.norm(rel_r))
    if rng_km <= 0.0:
        los = np.zeros(3, dtype=float)
        range_rate = 0.0
    else:
        los = rel_r / rng_km
        range_rate = float(np.dot(rel_v, los))
    az = float(np.arctan2(los[1], los[0])) if rng_km > 0.0 else 0.0
    el = float(np.arcsin(np.clip(los[2], -1.0, 1.0))) if rng_km > 0.0 else 0.0
    if normalized == "relative_range":
        return np.array([rng_km], dtype=float)
    if normalized == "relative_range_rate":
        return np.array([rng_km, range_rate], dtype=float)
    if normalized == "relative_angles":
        return np.array([az, el], dtype=float)
    if normalized == "relative_angles_range":
        return np.array([az, el, rng_km], dtype=float)
    if normalized == "relative_angles_range_rate":
        return np.array([az, el, rng_km, range_rate], dtype=float)
    raise ValueError(f"Unsupported HCW measurement_model '{model}'.")


def hcw_measurement_jacobian(model: str, state: np.ndarray, *, measurement_origin: str = "chief") -> np.ndarray:
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_state":
        sign = 1.0 if _normalize_measurement_origin(measurement_origin) == "chief" else -1.0
        return sign * np.eye(6)
    x = np.asarray(state, dtype=float).reshape(6)
    r = x[:3]
    v = x[3:]
    rho = float(np.linalg.norm(r))
    transverse = float(np.hypot(r[0], r[1]))
    if rho <= 1.0e-12:
        raise ValueError("Relative range/angle measurement Jacobian is singular at zero separation.")

    range_row = np.zeros(6, dtype=float)
    range_row[:3] = r / rho
    range_rate_row = np.zeros(6, dtype=float)
    radial_velocity = float(np.dot(r, v))
    range_rate_row[:3] = v / rho - radial_velocity * r / rho**3
    range_rate_row[3:] = r / rho
    if normalized == "relative_range":
        return range_row.reshape(1, 6)
    if normalized == "relative_range_rate":
        return np.vstack((range_row, range_rate_row))

    if transverse <= 1.0e-12:
        raise ValueError("Relative azimuth/elevation Jacobian is singular on the cross-track axis.")
    azimuth_row = np.zeros(6, dtype=float)
    azimuth_row[:3] = np.array([-r[1], r[0], 0.0]) / transverse**2
    elevation_row = np.zeros(6, dtype=float)
    elevation_row[:3] = np.array(
        [
            -r[0] * r[2] / (rho**2 * transverse),
            -r[1] * r[2] / (rho**2 * transverse),
            transverse / rho**2,
        ],
        dtype=float,
    )
    if _normalize_measurement_origin(measurement_origin) == "deputy":
        elevation_row *= -1.0
    if normalized == "relative_angles":
        return np.vstack((azimuth_row, elevation_row))
    if normalized == "relative_angles_range":
        return np.vstack((azimuth_row, elevation_row, range_row))
    if normalized == "relative_angles_range_rate":
        return np.vstack((azimuth_row, elevation_row, range_row, range_rate_row))
    raise ValueError(f"Unsupported HCW measurement_model '{model}'.")


def _diag6(value: np.ndarray, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"{field_name} must be length-6.")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return arr


def _normalize_numeric_backend(value: str) -> str:
    normalized = normalize_numeric_backend("python" if value is None else value, error_message="numeric_backend must be 'python' or 'rust'.")
    return normalized


def _validated_update_epochs(
    belief: StateBelief,
    measurement: Measurement | None,
    output_t_s: float,
) -> tuple[float, float]:
    """Validate the filter interval without silently moving a measurement epoch."""

    output_epoch = float(output_t_s)
    belief_epoch = float(belief.last_update_t_s)
    if not np.isfinite(output_epoch) or not np.isfinite(belief_epoch):
        raise ValueError("belief and output epochs must be finite.")
    if output_epoch < belief_epoch:
        raise ValueError("output epoch must not precede the current belief epoch.")
    if measurement is None:
        return output_epoch, output_epoch
    measurement_epoch = float(measurement.t_s)
    if not np.isfinite(measurement_epoch):
        raise ValueError("measurement epoch must be finite.")
    if measurement_epoch < belief_epoch:
        raise ValueError(
            "measurement epoch precedes the current belief epoch; out-of-sequence updates are not supported."
        )
    if measurement_epoch > output_epoch:
        raise ValueError("measurement epoch must not be later than the requested output epoch.")
    return output_epoch, measurement_epoch


def _measurement_covariance(value: np.ndarray, dimension: int) -> np.ndarray:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (dimension, dimension) or np.any(~np.isfinite(covariance)):
        raise ValueError(f"meas_noise_covariance must be a finite {dimension}x{dimension} matrix.")
    if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=1.0e-14):
        raise ValueError("meas_noise_covariance must be symmetric.")
    symmetric = 0.5 * (covariance + covariance.T)
    if np.min(np.linalg.eigvalsh(symmetric)) < -1.0e-14:
        raise ValueError("meas_noise_covariance must be positive semidefinite.")
    return symmetric


def _normalize_measurement_origin(value: str) -> str:
    raw = str(value or "chief").strip().lower().replace("-", "_")
    aliases = {
        "chief": "chief",
        "target": "chief",
        "deputy_from_chief": "chief",
        "observer": "deputy",
        "deputy": "deputy",
        "chaser": "deputy",
        "chief_from_deputy": "deputy",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"chief", "deputy"}:
        raise ValueError("measurement_origin must be 'chief' or 'deputy'.")
    return normalized


def _default_measurement_noise_for_model(model: str, full_state_diag: np.ndarray) -> np.ndarray:
    pos_var = float(np.mean(np.asarray(full_state_diag[:3], dtype=float)))
    vel_var = float(np.mean(np.asarray(full_state_diag[3:6], dtype=float)))
    normalized = normalize_hcw_measurement_model(model)
    if normalized == "relative_range":
        return np.array([pos_var], dtype=float)
    if normalized == "relative_range_rate":
        return np.array([pos_var, vel_var], dtype=float)
    if normalized in {"relative_angles", "relative_angles_range", "relative_angles_range_rate"}:
        raise ValueError(
            "Angular measurement covariance must be supplied in rad^2 with the exact measurement-model "
            "dimension; Cartesian position variance cannot be converted to angle variance without range geometry."
        )
    return np.asarray(full_state_diag, dtype=float).reshape(6)


def _measurement_innovation(model: str, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    innovation = np.asarray(z, dtype=float).reshape(-1) - np.asarray(h, dtype=float).reshape(-1)
    normalized = normalize_hcw_measurement_model(model)
    if normalized in {"relative_angles", "relative_angles_range", "relative_angles_range_rate"}:
        innovation[0] = _wrap_angle_rad(float(innovation[0]))
        innovation[1] = _wrap_angle_rad(float(innovation[1]))
    return innovation


def _wrap_angle_rad(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)
