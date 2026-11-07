from __future__ import annotations

from dataclasses import dataclass
from math import acos, asin, cos, degrees, exp, log, pi, radians, sin, sqrt

import numpy as np

from sim.dynamics.orbit.atmosphere import density_ussa1976
from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_J2, EARTH_MU_KM3_S2, EARTH_RADIUS_KM

STANDARD_GRAVITY_M_S2 = 9.80665
SIDEREAL_DAY_S = 86164.0905
EARTH_ROTATION_RATE_RAD_S = 2.0 * pi / SIDEREAL_DAY_S
TROPICAL_YEAR_DAYS = 365.2422
USSA1976_MAX_ALTITUDE_KM = 1000.0
ECCENTRICITY_SINGULAR_TOL = 1.0e-8
INCLINATION_SINGULAR_TOL_DEG = 1.0e-8
ENERGY_PARABOLIC_TOL = 1.0e-12


def _finite(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _positive(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_int(value: float, name: str) -> int:
    numeric = _positive(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _wrap_degrees(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def _wrap_signed_degrees(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _angle_degrees_from_components(y_component: float, x_component: float) -> float:
    return _wrap_degrees(degrees(float(np.arctan2(y_component, x_component))))


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    array = np.array(values, dtype=float).reshape(3)
    return (float(array[0]), float(array[1]), float(array[2]))


def _require_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.array(values, dtype=float).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


@dataclass(frozen=True)
class CircularOrbitResult:
    altitude_km: float
    radius_km: float
    velocity_km_s: float
    period_s: float
    mean_motion_rad_s: float
    escape_velocity_km_s: float

    @property
    def period_min(self) -> float:
        return self.period_s / 60.0


@dataclass(frozen=True)
class OrbitalPeriodResult:
    semi_major_axis_km: float
    altitude_km: float | None
    period_s: float
    mean_motion_rad_s: float

    @property
    def period_min(self) -> float:
        return self.period_s / 60.0


@dataclass(frozen=True)
class AltitudeFromPeriodResult:
    period_s: float
    semi_major_axis_km: float
    altitude_km: float
    mean_motion_rad_s: float

    @property
    def period_min(self) -> float:
        return self.period_s / 60.0


@dataclass(frozen=True)
class VisVivaResult:
    radius_km: float
    semi_major_axis_km: float
    velocity_km_s: float


@dataclass(frozen=True)
class HohmannTransferResult:
    initial_altitude_km: float
    final_altitude_km: float
    initial_radius_km: float
    final_radius_km: float
    transfer_semimajor_axis_km: float
    first_burn_delta_v_km_s: float
    second_burn_delta_v_km_s: float
    total_delta_v_km_s: float
    transfer_time_s: float

    @property
    def first_burn_delta_v_m_s(self) -> float:
        return self.first_burn_delta_v_km_s * 1000.0

    @property
    def second_burn_delta_v_m_s(self) -> float:
        return self.second_burn_delta_v_km_s * 1000.0

    @property
    def total_delta_v_m_s(self) -> float:
        return self.total_delta_v_km_s * 1000.0

    @property
    def transfer_time_min(self) -> float:
        return self.transfer_time_s / 60.0


@dataclass(frozen=True)
class HohmannRendezvousPhaseResult:
    initial_altitude_km: float
    target_altitude_km: float
    initial_radius_km: float
    target_radius_km: float
    transfer_semimajor_axis_km: float
    transfer_time_s: float
    initial_mean_motion_rad_s: float
    target_mean_motion_rad_s: float
    relative_phase_rate_deg_s: float
    required_phase_angle_deg: float
    target_travel_during_transfer_deg: float
    synodic_period_s: float | None

    @property
    def transfer_time_min(self) -> float:
        return self.transfer_time_s / 60.0

    @property
    def synodic_period_min(self) -> float | None:
        return None if self.synodic_period_s is None else self.synodic_period_s / 60.0


@dataclass(frozen=True)
class HohmannRendezvousWaitTimeResult:
    initial_altitude_km: float
    target_altitude_km: float
    current_phase_angle_deg: float
    required_phase_angle_deg: float
    relative_phase_rate_deg_s: float
    wait_time_s: float | None
    synodic_period_s: float | None

    @property
    def wait_time_min(self) -> float | None:
        return None if self.wait_time_s is None else self.wait_time_s / 60.0

    @property
    def wait_time_hr(self) -> float | None:
        return None if self.wait_time_s is None else self.wait_time_s / 3600.0

    @property
    def synodic_period_min(self) -> float | None:
        return None if self.synodic_period_s is None else self.synodic_period_s / 60.0


@dataclass(frozen=True)
class PlaneChangeResult:
    speed_km_s: float
    angle_deg: float
    delta_v_km_s: float

    @property
    def delta_v_m_s(self) -> float:
        return self.delta_v_km_s * 1000.0


@dataclass(frozen=True)
class EscapeVelocityResult:
    radius_km: float
    altitude_km: float | None
    escape_velocity_km_s: float


@dataclass(frozen=True)
class GeosynchronousOrbitResult:
    period_s: float
    semi_major_axis_km: float
    altitude_km: float
    circular_velocity_km_s: float
    mean_motion_rad_s: float

    @property
    def period_hr(self) -> float:
        return self.period_s / 3600.0


@dataclass(frozen=True)
class ApogeePerigeeFromElementsResult:
    semi_major_axis_km: float
    eccentricity: float
    perigee_radius_km: float
    apogee_radius_km: float
    perigee_altitude_km: float
    apogee_altitude_km: float


@dataclass(frozen=True)
class ElementsFromApogeePerigeeResult:
    perigee_altitude_km: float
    apogee_altitude_km: float
    perigee_radius_km: float
    apogee_radius_km: float
    semi_major_axis_km: float
    eccentricity: float


@dataclass(frozen=True)
class ApogeePerigeeVelocitiesResult:
    perigee_altitude_km: float
    apogee_altitude_km: float
    semi_major_axis_km: float
    eccentricity: float
    perigee_velocity_km_s: float
    apogee_velocity_km_s: float


@dataclass(frozen=True)
class CombinedPlaneChangeResult:
    initial_speed_km_s: float
    final_speed_km_s: float
    angle_deg: float
    delta_v_km_s: float

    @property
    def delta_v_m_s(self) -> float:
        return self.delta_v_km_s * 1000.0


@dataclass(frozen=True)
class SunSynchronousInclinationResult:
    altitude_km: float
    semi_major_axis_km: float
    inclination_deg: float
    nodal_precession_deg_day: float
    target_precession_deg_day: float


@dataclass(frozen=True)
class J2SecularRatesResult:
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    semi_latus_rectum_km: float
    mean_motion_rad_s: float
    orbital_period_s: float
    raan_rate_deg_day: float
    argument_of_perigee_rate_deg_day: float
    mean_anomaly_j2_rate_deg_day: float

    @property
    def orbital_period_min(self) -> float:
        return self.orbital_period_s / 60.0


@dataclass(frozen=True)
class HCWNaturalMotionResult:
    altitude_km: float
    orbit_radius_km: float
    mean_motion_rad_s: float
    natural_motion_period_s: float
    circular_velocity_km_s: float

    @property
    def natural_motion_period_min(self) -> float:
        return self.natural_motion_period_s / 60.0


@dataclass(frozen=True)
class HCWDriftEstimateResult:
    altitude_km: float
    duration_orbits: float
    duration_s: float
    radial_offset_m: float
    intrack_velocity_bias_m_s: float
    radial_offset_drift_m: float
    intrack_velocity_drift_m: float
    total_intrack_drift_m: float
    mean_motion_rad_s: float

    @property
    def duration_min(self) -> float:
        return self.duration_s / 60.0


@dataclass(frozen=True)
class EclipseEstimateResult:
    altitude_km: float
    beta_angle_deg: float
    orbit_radius_km: float
    orbital_period_s: float
    beta_critical_deg: float
    eclipse_half_angle_deg: float | None
    eclipse_duration_s: float
    eclipse_fraction: float
    sunlight_duration_s: float

    @property
    def orbital_period_min(self) -> float:
        return self.orbital_period_s / 60.0

    @property
    def eclipse_duration_min(self) -> float:
        return self.eclipse_duration_s / 60.0

    @property
    def sunlight_duration_min(self) -> float:
        return self.sunlight_duration_s / 60.0


@dataclass(frozen=True)
class GroundTrackDriftResult:
    altitude_km: float
    orbital_period_s: float
    earth_rotation_deg_per_orbit: float
    westward_drift_deg_per_orbit: float
    orbits_per_sidereal_day: float
    equator_drift_km_per_orbit: float

    @property
    def orbital_period_min(self) -> float:
        return self.orbital_period_s / 60.0


@dataclass(frozen=True)
class RepeatGroundTrackResult:
    altitude_km: float
    max_days: int
    repeat_days: int
    repeat_orbits: int
    ground_track_error_deg: float
    ground_track_error_km: float
    exact_repeat_altitude_km: float
    exact_repeat_period_s: float

    @property
    def exact_repeat_period_min(self) -> float:
        return self.exact_repeat_period_s / 60.0


@dataclass(frozen=True)
class EntryInterfaceEstimateResult:
    apogee_altitude_km: float
    perigee_altitude_km: float
    interface_altitude_km: float
    semi_major_axis_km: float
    eccentricity: float
    interface_radius_km: float
    speed_km_s: float
    flight_path_angle_deg: float
    true_anomaly_deg: float
    note: str


@dataclass(frozen=True)
class PhasingDriftResult:
    reference_altitude_km: float
    phasing_altitude_km: float
    altitude_change_km: float
    reference_mean_motion_rad_s: float
    phasing_mean_motion_rad_s: float
    drift_rate_m_s: float
    drift_per_reference_orbit_km: float
    lap_time_s: float | None

    @property
    def drift_rate_km_s(self) -> float:
        return self.drift_rate_m_s / 1000.0

    @property
    def lap_time_hr(self) -> float | None:
        return None if self.lap_time_s is None else self.lap_time_s / 3600.0


@dataclass(frozen=True)
class MissionRecoveryIntrackImpulseResult:
    reference_altitude_km: float
    reference_radius_km: float
    reference_period_s: float
    circular_speed_km_s: float
    disturbance_delta_v_m_s: float
    disturbed_speed_km_s: float
    disturbed_semi_major_axis_km: float
    disturbed_eccentricity: float
    disturbed_perigee_altitude_km: float
    disturbed_apogee_altitude_km: float
    disturbed_period_s: float
    disturbance_apsis: str
    recovery_delta_v_m_s: float
    recovery_propellant_kg: float
    recovery_propellant_fraction: float
    total_event_delta_v_m_s: float
    total_event_propellant_fraction: float
    spacecraft_mass_kg: float
    isp_s: float
    slot_tolerance_deg: float
    max_phasing_orbits: int
    continuous_slot_lap_time_s: float | None
    slot_recovery_found: bool
    slot_recovery_orbits: int | None
    slot_recovery_time_s: float | None
    slot_recovery_phase_error_deg: float | None
    best_slot_orbits: int
    best_slot_time_s: float
    best_slot_phase_error_deg: float
    notes: tuple[str, ...]

    @property
    def reference_period_min(self) -> float:
        return self.reference_period_s / 60.0

    @property
    def disturbed_period_min(self) -> float:
        return self.disturbed_period_s / 60.0

    @property
    def continuous_slot_lap_time_hr(self) -> float | None:
        return None if self.continuous_slot_lap_time_s is None else self.continuous_slot_lap_time_s / 3600.0

    @property
    def slot_recovery_time_hr(self) -> float | None:
        return None if self.slot_recovery_time_s is None else self.slot_recovery_time_s / 3600.0

    @property
    def best_slot_time_hr(self) -> float:
        return self.best_slot_time_s / 3600.0


@dataclass(frozen=True)
class RocketEquationDeltaVResult:
    isp_s: float
    mass_ratio: float
    delta_v_m_s: float

    @property
    def delta_v_km_s(self) -> float:
        return self.delta_v_m_s / 1000.0


@dataclass(frozen=True)
class RocketEquationMassRatioResult:
    isp_s: float
    delta_v_m_s: float
    mass_ratio: float
    propellant_fraction: float


@dataclass(frozen=True)
class BallisticCoefficientResult:
    mass_kg: float
    drag_coefficient: float
    drag_area_m2: float
    ballistic_coefficient_kg_m2: float


@dataclass(frozen=True)
class AtmosphericDensityResult:
    altitude_km: float
    density_kg_m3: float
    warning: str


@dataclass(frozen=True)
class DragForceResult:
    density_kg_m3: float
    speed_m_s: float
    drag_coefficient: float
    drag_area_m2: float
    mass_kg: float
    drag_force_n: float
    drag_accel_m_s2: float
    ballistic_coefficient_kg_m2: float


@dataclass(frozen=True)
class DragAtAltitudeResult:
    altitude_km: float
    density_kg_m3: float
    circular_speed_m_s: float
    drag_coefficient: float
    drag_area_m2: float
    mass_kg: float
    drag_force_n: float
    drag_accel_m_s2: float
    ballistic_coefficient_kg_m2: float
    warning: str


@dataclass(frozen=True)
class CircularOrbitDecayRateResult:
    altitude_km: float
    density_kg_m3: float
    ballistic_coefficient_kg_m2: float
    decay_rate_m_s: float
    decay_rate_m_day: float
    decay_rate_km_day: float
    warning: str


@dataclass(frozen=True)
class DragLifetimeRangeResult:
    initial_altitude_km: float
    deorbit_altitude_km: float
    ballistic_coefficient_kg_m2: float
    low_drag_density_scale: float
    nominal_density_scale: float
    high_drag_density_scale: float
    low_drag_lifetime_s: float
    nominal_lifetime_s: float
    high_drag_lifetime_s: float
    integration_step_km: float
    warning: str

    @property
    def low_drag_lifetime_days(self) -> float:
        return self.low_drag_lifetime_s / 86400.0

    @property
    def nominal_lifetime_days(self) -> float:
        return self.nominal_lifetime_s / 86400.0

    @property
    def high_drag_lifetime_days(self) -> float:
        return self.high_drag_lifetime_s / 86400.0

    @property
    def low_drag_lifetime_years(self) -> float:
        return self.low_drag_lifetime_days / 365.25

    @property
    def nominal_lifetime_years(self) -> float:
        return self.nominal_lifetime_days / 365.25

    @property
    def high_drag_lifetime_years(self) -> float:
        return self.high_drag_lifetime_days / 365.25


@dataclass(frozen=True)
class RVToElementsResult:
    position_eci_km: tuple[float, float, float]
    velocity_eci_km_s: tuple[float, float, float]
    radius_km: float
    speed_km_s: float
    specific_angular_momentum_km2_s: float
    specific_energy_km2_s2: float
    semi_major_axis_km: float | None
    semi_latus_rectum_km: float
    eccentricity: float
    perigee_radius_km: float
    perigee_altitude_km: float
    apogee_radius_km: float | None
    apogee_altitude_km: float | None
    period_s: float | None
    orbit_type: str
    inclination_deg: float
    raan_deg: float | None
    argp_deg: float | None
    true_anomaly_deg: float | None
    argument_of_latitude_deg: float | None
    longitude_of_perigee_deg: float | None
    true_longitude_deg: float | None
    v_infinity_km_s: float | None
    turning_angle_deg: float | None
    notes: tuple[str, ...]

    @property
    def period_min(self) -> float | None:
        return None if self.period_s is None else self.period_s / 60.0

@dataclass(frozen=True)
class COEToRVResult:
    input_mode: str
    position_eci_km: tuple[float, float, float]
    velocity_eci_km_s: tuple[float, float, float]
    radius_km: float
    speed_km_s: float
    semi_major_axis_km: float | None
    eccentricity: float
    inclination_deg: float
    raan_deg: float | None
    argp_deg: float | None
    true_anomaly_deg: float | None
    argument_of_latitude_deg: float | None
    longitude_of_perigee_deg: float | None
    true_longitude_deg: float | None
    perigee_altitude_km: float
    apogee_altitude_km: float | None
    period_s: float | None
    notes: tuple[str, ...]

    @property
    def period_min(self) -> float | None:
        return None if self.period_s is None else self.period_s / 60.0

def rv_to_robust_elements(
    rx_km: float,
    ry_km: float,
    rz_km: float,
    vx_km_s: float,
    vy_km_s: float,
    vz_km_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> RVToElementsResult:
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    r = _require_finite_vector(np.array([rx_km, ry_km, rz_km], dtype=float), "position vector")
    v = _require_finite_vector(np.array([vx_km_s, vy_km_s, vz_km_s], dtype=float), "velocity vector")
    radius = _positive(float(np.linalg.norm(r)), "position norm")
    speed = float(np.linalg.norm(v))

    h_vec = np.cross(r, v)
    h_norm = _positive(float(np.linalg.norm(h_vec)), "angular momentum norm")
    h_hat = h_vec / h_norm
    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.cross(k_hat, h_vec)
    n_norm = float(np.linalg.norm(n_vec))
    e_vec = (np.cross(v, h_vec) / mu) - (r / radius)
    eccentricity = float(np.linalg.norm(e_vec))
    energy = 0.5 * speed * speed - mu / radius
    p = h_norm * h_norm / mu

    semi_major_axis = None if abs(energy) <= ENERGY_PARABOLIC_TOL else -mu / (2.0 * energy)
    if eccentricity < ECCENTRICITY_SINGULAR_TOL:
        orbit_type = "circular"
    elif abs(eccentricity - 1.0) <= 1.0e-6 or semi_major_axis is None:
        orbit_type = "parabolic"
    elif eccentricity < 1.0:
        orbit_type = "elliptical"
    else:
        orbit_type = "hyperbolic"

    perigee_radius = p / (1.0 + eccentricity)
    apogee_radius = p / (1.0 - eccentricity) if eccentricity < 1.0 else None
    period_s = None
    if semi_major_axis is not None and semi_major_axis > 0.0 and eccentricity < 1.0:
        period_s = 2.0 * pi * sqrt((semi_major_axis**3) / mu)

    inclination = degrees(acos(float(np.clip(h_vec[2] / h_norm, -1.0, 1.0))))
    is_equatorial = inclination <= INCLINATION_SINGULAR_TOL_DEG or abs(inclination - 180.0) <= INCLINATION_SINGULAR_TOL_DEG
    is_circular = eccentricity < ECCENTRICITY_SINGULAR_TOL

    raan = None
    if not is_equatorial and n_norm > 0.0:
        raan = _angle_degrees_from_components(n_vec[1], n_vec[0])

    argp = None
    if not is_equatorial and not is_circular:
        argp = _angle_degrees_from_components(
            float(np.dot(np.cross(n_vec, e_vec), h_hat)),
            float(np.dot(n_vec, e_vec)),
        )

    true_anomaly = None
    if not is_circular:
        true_anomaly = _angle_degrees_from_components(
            float(np.dot(np.cross(e_vec, r), h_hat)),
            float(np.dot(e_vec, r)),
        )

    argument_of_latitude = None
    if is_circular and not is_equatorial:
        argument_of_latitude = _angle_degrees_from_components(
            float(np.dot(np.cross(n_vec, r), h_hat)),
            float(np.dot(n_vec, r)),
        )

    longitude_of_perigee = None
    if is_equatorial and not is_circular:
        longitude_of_perigee = _angle_degrees_from_components(e_vec[1], e_vec[0])

    true_longitude = None
    if is_equatorial and is_circular:
        true_longitude = _angle_degrees_from_components(r[1], r[0])

    v_infinity = None
    turning_angle = None
    if eccentricity > 1.0 and energy > 0.0:
        v_infinity = sqrt(2.0 * energy)
        turning_angle = degrees(2.0 * float(np.arcsin(1.0 / eccentricity)))

    notes: list[str] = []
    if is_circular:
        notes.append("Circular orbit: argument of perigee and true anomaly are undefined.")
    if is_equatorial:
        notes.append("Equatorial orbit: RAAN is undefined.")
    if is_circular and not is_equatorial:
        notes.append("Use argument of latitude instead of argument of perigee plus true anomaly.")
    if is_equatorial and not is_circular:
        notes.append("Use longitude of perigee instead of RAAN plus argument of perigee.")
    if is_equatorial and is_circular:
        notes.append("Use true longitude instead of RAAN, argument of perigee, and true anomaly.")
    if orbit_type in {"parabolic", "hyperbolic"}:
        notes.append("Non-elliptical trajectory: period and apogee are not defined.")

    return RVToElementsResult(
        position_eci_km=_tuple3(r),
        velocity_eci_km_s=_tuple3(v),
        radius_km=radius,
        speed_km_s=speed,
        specific_angular_momentum_km2_s=h_norm,
        specific_energy_km2_s2=energy,
        semi_major_axis_km=None if semi_major_axis is None else float(semi_major_axis),
        semi_latus_rectum_km=p,
        eccentricity=eccentricity,
        perigee_radius_km=perigee_radius,
        perigee_altitude_km=perigee_radius - body_radius,
        apogee_radius_km=apogee_radius,
        apogee_altitude_km=None if apogee_radius is None else apogee_radius - body_radius,
        period_s=period_s,
        orbit_type=orbit_type,
        inclination_deg=inclination,
        raan_deg=raan,
        argp_deg=argp,
        true_anomaly_deg=true_anomaly,
        argument_of_latitude_deg=argument_of_latitude,
        longitude_of_perigee_deg=longitude_of_perigee,
        true_longitude_deg=true_longitude,
        v_infinity_km_s=v_infinity,
        turning_angle_deg=turning_angle,
        notes=tuple(notes),
    )


def _coe_to_rv_result(
    *,
    input_mode: str,
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_km3_s2: float,
    body_radius_km: float,
) -> COEToRVResult:
    r_eci_km, v_eci_km_s = coe_to_rv_eci(
        a_km=semi_major_axis_km,
        ecc=eccentricity,
        inc_deg=inclination_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_deg,
        mu_km3_s2=mu_km3_s2,
    )
    elements = rv_to_robust_elements(
        float(r_eci_km[0]),
        float(r_eci_km[1]),
        float(r_eci_km[2]),
        float(v_eci_km_s[0]),
        float(v_eci_km_s[1]),
        float(v_eci_km_s[2]),
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )
    return COEToRVResult(
        input_mode=input_mode,
        position_eci_km=elements.position_eci_km,
        velocity_eci_km_s=elements.velocity_eci_km_s,
        radius_km=elements.radius_km,
        speed_km_s=elements.speed_km_s,
        semi_major_axis_km=elements.semi_major_axis_km,
        eccentricity=elements.eccentricity,
        inclination_deg=elements.inclination_deg,
        raan_deg=elements.raan_deg,
        argp_deg=elements.argp_deg,
        true_anomaly_deg=elements.true_anomaly_deg,
        argument_of_latitude_deg=elements.argument_of_latitude_deg,
        longitude_of_perigee_deg=elements.longitude_of_perigee_deg,
        true_longitude_deg=elements.true_longitude_deg,
        perigee_altitude_km=elements.perigee_altitude_km,
        apogee_altitude_km=elements.apogee_altitude_km,
        period_s=elements.period_s,
        notes=elements.notes,
    )


def classical_coe_to_rv(
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> COEToRVResult:
    return _coe_to_rv_result(
        input_mode="Classical COE",
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_deg,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )


def circular_inclined_elements_to_rv(
    semi_major_axis_km: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_latitude_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> COEToRVResult:
    return _coe_to_rv_result(
        input_mode="Circular inclined alternate elements",
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=0.0,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        argp_deg=0.0,
        true_anomaly_deg=argument_of_latitude_deg,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )


def equatorial_elliptical_elements_to_rv(
    semi_major_axis_km: float,
    eccentricity: float,
    longitude_of_perigee_deg: float,
    true_anomaly_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> COEToRVResult:
    return _coe_to_rv_result(
        input_mode="Equatorial elliptical alternate elements",
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=eccentricity,
        inclination_deg=0.0,
        raan_deg=0.0,
        argp_deg=longitude_of_perigee_deg,
        true_anomaly_deg=true_anomaly_deg,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )


def circular_equatorial_elements_to_rv(
    semi_major_axis_km: float,
    true_longitude_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> COEToRVResult:
    return _coe_to_rv_result(
        input_mode="Circular equatorial alternate elements",
        semi_major_axis_km=semi_major_axis_km,
        eccentricity=0.0,
        inclination_deg=0.0,
        raan_deg=0.0,
        argp_deg=0.0,
        true_anomaly_deg=true_longitude_deg,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )


def circular_orbit_from_altitude(
    altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> CircularOrbitResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    body_radius = _positive(body_radius_km, "body_radius_km")
    radius = _positive(body_radius + altitude, "orbit radius")
    return circular_orbit_from_radius(radius, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius)


def circular_orbit_from_radius(
    radius_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> CircularOrbitResult:
    radius = _positive(radius_km, "radius_km")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    velocity = sqrt(mu / radius)
    mean_motion = sqrt(mu / (radius**3))
    period = 2.0 * pi / mean_motion
    return CircularOrbitResult(
        altitude_km=radius - body_radius,
        radius_km=radius,
        velocity_km_s=velocity,
        period_s=period,
        mean_motion_rad_s=mean_motion,
        escape_velocity_km_s=escape_velocity_at_radius(radius, mu_km3_s2=mu).escape_velocity_km_s,
    )


def orbital_period_from_altitude(
    altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> OrbitalPeriodResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    body_radius = _positive(body_radius_km, "body_radius_km")
    return orbital_period_from_semimajor_axis(
        body_radius + altitude,
        altitude_km=altitude,
        mu_km3_s2=mu_km3_s2,
    )


def orbital_period_from_semimajor_axis(
    semi_major_axis_km: float,
    *,
    altitude_km: float | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> OrbitalPeriodResult:
    a = _positive(semi_major_axis_km, "semi_major_axis_km")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    mean_motion = sqrt(mu / (a**3))
    period = 2.0 * pi / mean_motion
    return OrbitalPeriodResult(
        semi_major_axis_km=a,
        altitude_km=None if altitude_km is None else float(altitude_km),
        period_s=period,
        mean_motion_rad_s=mean_motion,
    )


def altitude_from_period(
    period_s: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> AltitudeFromPeriodResult:
    period = _positive(period_s, "period_s")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    mean_motion = 2.0 * pi / period
    semi_major_axis = (mu / (mean_motion**2)) ** (1.0 / 3.0)
    return AltitudeFromPeriodResult(
        period_s=period,
        semi_major_axis_km=semi_major_axis,
        altitude_km=semi_major_axis - body_radius,
        mean_motion_rad_s=mean_motion,
    )


def vis_viva_velocity(
    radius_km: float,
    semi_major_axis_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> VisVivaResult:
    radius = _positive(radius_km, "radius_km")
    a = _positive(semi_major_axis_km, "semi_major_axis_km")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    speed_squared = mu * ((2.0 / radius) - (1.0 / a))
    if speed_squared < 0.0:
        raise ValueError("radius_km and semi_major_axis_km describe an impossible elliptical speed.")
    return VisVivaResult(radius_km=radius, semi_major_axis_km=a, velocity_km_s=sqrt(speed_squared))


def hohmann_transfer_between_circular_orbits(
    initial_altitude_km: float,
    final_altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> HohmannTransferResult:
    initial_altitude = _nonnegative(initial_altitude_km, "initial_altitude_km")
    final_altitude = _nonnegative(final_altitude_km, "final_altitude_km")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    r1 = _positive(body_radius + initial_altitude, "initial radius")
    r2 = _positive(body_radius + final_altitude, "final radius")
    if r1 == r2:
        raise ValueError("initial_altitude_km and final_altitude_km must differ for a Hohmann transfer.")

    transfer_a = 0.5 * (r1 + r2)
    initial_circular_speed = sqrt(mu / r1)
    final_circular_speed = sqrt(mu / r2)
    transfer_speed_at_initial = vis_viva_velocity(r1, transfer_a, mu_km3_s2=mu).velocity_km_s
    transfer_speed_at_final = vis_viva_velocity(r2, transfer_a, mu_km3_s2=mu).velocity_km_s
    first_burn = abs(transfer_speed_at_initial - initial_circular_speed)
    second_burn = abs(final_circular_speed - transfer_speed_at_final)
    transfer_time = pi * sqrt((transfer_a**3) / mu)
    return HohmannTransferResult(
        initial_altitude_km=initial_altitude,
        final_altitude_km=final_altitude,
        initial_radius_km=r1,
        final_radius_km=r2,
        transfer_semimajor_axis_km=transfer_a,
        first_burn_delta_v_km_s=first_burn,
        second_burn_delta_v_km_s=second_burn,
        total_delta_v_km_s=first_burn + second_burn,
        transfer_time_s=transfer_time,
    )


def hohmann_rendezvous_phase_angle(
    initial_altitude_km: float,
    target_altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> HohmannRendezvousPhaseResult:
    initial_altitude = _nonnegative(initial_altitude_km, "initial_altitude_km")
    target_altitude = _nonnegative(target_altitude_km, "target_altitude_km")
    if initial_altitude == target_altitude:
        raise ValueError("initial_altitude_km and target_altitude_km must differ for rendezvous phasing.")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    initial_radius = body_radius + initial_altitude
    target_radius = body_radius + target_altitude
    transfer_a = 0.5 * (initial_radius + target_radius)
    transfer_time = pi * sqrt((transfer_a**3) / mu)
    initial_n = sqrt(mu / (initial_radius**3))
    target_n = sqrt(mu / (target_radius**3))
    target_travel_deg = degrees(target_n * transfer_time)
    required_phase = _wrap_signed_degrees(180.0 - target_travel_deg)
    relative_rate_deg_s = degrees(target_n - initial_n)
    synodic_period = None if abs(relative_rate_deg_s) <= 1.0e-15 else 360.0 / abs(relative_rate_deg_s)
    return HohmannRendezvousPhaseResult(
        initial_altitude_km=initial_altitude,
        target_altitude_km=target_altitude,
        initial_radius_km=initial_radius,
        target_radius_km=target_radius,
        transfer_semimajor_axis_km=transfer_a,
        transfer_time_s=transfer_time,
        initial_mean_motion_rad_s=initial_n,
        target_mean_motion_rad_s=target_n,
        relative_phase_rate_deg_s=relative_rate_deg_s,
        required_phase_angle_deg=required_phase,
        target_travel_during_transfer_deg=target_travel_deg,
        synodic_period_s=synodic_period,
    )


def hohmann_rendezvous_wait_time(
    initial_altitude_km: float,
    target_altitude_km: float,
    current_phase_angle_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> HohmannRendezvousWaitTimeResult:
    phase = hohmann_rendezvous_phase_angle(
        initial_altitude_km,
        target_altitude_km,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
    )
    current_phase = _wrap_signed_degrees(current_phase_angle_deg)
    if abs(phase.relative_phase_rate_deg_s) <= 1.0e-15:
        wait_time = None
    elif phase.relative_phase_rate_deg_s > 0.0:
        delta = (phase.required_phase_angle_deg - current_phase) % 360.0
        wait_time = delta / phase.relative_phase_rate_deg_s
    else:
        delta = -((current_phase - phase.required_phase_angle_deg) % 360.0)
        wait_time = delta / phase.relative_phase_rate_deg_s
    return HohmannRendezvousWaitTimeResult(
        initial_altitude_km=phase.initial_altitude_km,
        target_altitude_km=phase.target_altitude_km,
        current_phase_angle_deg=current_phase,
        required_phase_angle_deg=phase.required_phase_angle_deg,
        relative_phase_rate_deg_s=phase.relative_phase_rate_deg_s,
        wait_time_s=wait_time,
        synodic_period_s=phase.synodic_period_s,
    )


def plane_change_delta_v(speed_km_s: float, angle_deg: float) -> PlaneChangeResult:
    speed = _nonnegative(speed_km_s, "speed_km_s")
    angle = _nonnegative(angle_deg, "angle_deg")
    if angle > 180.0:
        raise ValueError("angle_deg must be no greater than 180.")
    delta_v = sqrt((2.0 * speed * speed) * (1.0 - cos(radians(angle))))
    return PlaneChangeResult(speed_km_s=speed, angle_deg=angle, delta_v_km_s=delta_v)


def escape_velocity_at_radius(
    radius_km: float,
    *,
    altitude_km: float | None = None,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> EscapeVelocityResult:
    radius = _positive(radius_km, "radius_km")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    return EscapeVelocityResult(radius_km=radius, altitude_km=altitude_km, escape_velocity_km_s=sqrt(2.0 * mu / radius))


def geosynchronous_orbit(
    *,
    period_s: float = SIDEREAL_DAY_S,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> GeosynchronousOrbitResult:
    period = _positive(period_s, "period_s")
    circular = altitude_from_period(period, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    speed = circular_orbit_from_radius(circular.semi_major_axis_km, mu_km3_s2=mu_km3_s2).velocity_km_s
    return GeosynchronousOrbitResult(
        period_s=period,
        semi_major_axis_km=circular.semi_major_axis_km,
        altitude_km=circular.altitude_km,
        circular_velocity_km_s=speed,
        mean_motion_rad_s=circular.mean_motion_rad_s,
    )


def apogee_perigee_from_elements(
    semi_major_axis_km: float,
    eccentricity: float,
    *,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> ApogeePerigeeFromElementsResult:
    a = _positive(semi_major_axis_km, "semi_major_axis_km")
    e = _nonnegative(eccentricity, "eccentricity")
    body_radius = _positive(body_radius_km, "body_radius_km")
    if e >= 1.0:
        raise ValueError("eccentricity must be less than 1 for an elliptical orbit.")
    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    return ApogeePerigeeFromElementsResult(
        semi_major_axis_km=a,
        eccentricity=e,
        perigee_radius_km=rp,
        apogee_radius_km=ra,
        perigee_altitude_km=rp - body_radius,
        apogee_altitude_km=ra - body_radius,
    )

def elements_from_apogee_perigee_altitudes(
    perigee_altitude_km: float,
    apogee_altitude_km: float,
    *,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> ElementsFromApogeePerigeeResult:
    perigee_altitude = _nonnegative(perigee_altitude_km, "perigee_altitude_km")
    apogee_altitude = _nonnegative(apogee_altitude_km, "apogee_altitude_km")
    body_radius = _positive(body_radius_km, "body_radius_km")
    if apogee_altitude < perigee_altitude:
        raise ValueError("apogee_altitude_km must be greater than or equal to perigee_altitude_km.")
    rp = body_radius + perigee_altitude
    ra = body_radius + apogee_altitude
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    return ElementsFromApogeePerigeeResult(
        perigee_altitude_km=perigee_altitude,
        apogee_altitude_km=apogee_altitude,
        perigee_radius_km=rp,
        apogee_radius_km=ra,
        semi_major_axis_km=a,
        eccentricity=e,
    )


def apogee_perigee_velocities_from_altitudes(
    perigee_altitude_km: float,
    apogee_altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> ApogeePerigeeVelocitiesResult:
    elements = elements_from_apogee_perigee_altitudes(
        perigee_altitude_km=perigee_altitude_km,
        apogee_altitude_km=apogee_altitude_km,
        body_radius_km=body_radius_km,
    )
    vp = vis_viva_velocity(elements.perigee_radius_km, elements.semi_major_axis_km, mu_km3_s2=mu_km3_s2)
    va = vis_viva_velocity(elements.apogee_radius_km, elements.semi_major_axis_km, mu_km3_s2=mu_km3_s2)
    return ApogeePerigeeVelocitiesResult(
        perigee_altitude_km=elements.perigee_altitude_km,
        apogee_altitude_km=elements.apogee_altitude_km,
        semi_major_axis_km=elements.semi_major_axis_km,
        eccentricity=elements.eccentricity,
        perigee_velocity_km_s=vp.velocity_km_s,
        apogee_velocity_km_s=va.velocity_km_s,
    )


def inclination_change_from_altitude(
    altitude_km: float,
    angle_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> PlaneChangeResult:
    circular = circular_orbit_from_altitude(altitude_km, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    return plane_change_delta_v(circular.velocity_km_s, angle_deg)


def combined_speed_plane_change_delta_v(
    initial_speed_km_s: float,
    final_speed_km_s: float,
    angle_deg: float,
) -> CombinedPlaneChangeResult:
    v1 = _nonnegative(initial_speed_km_s, "initial_speed_km_s")
    v2 = _nonnegative(final_speed_km_s, "final_speed_km_s")
    angle = _nonnegative(angle_deg, "angle_deg")
    if angle > 180.0:
        raise ValueError("angle_deg must be no greater than 180.")
    delta_v = sqrt(v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * cos(radians(angle)))
    return CombinedPlaneChangeResult(initial_speed_km_s=v1, final_speed_km_s=v2, angle_deg=angle, delta_v_km_s=delta_v)


def sun_synchronous_inclination_from_altitude(
    altitude_km: float,
    *,
    eccentricity: float = 0.0,
    target_precession_deg_day: float = 360.0 / TROPICAL_YEAR_DAYS,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
    j2: float = EARTH_J2,
) -> SunSynchronousInclinationResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    e = _nonnegative(eccentricity, "eccentricity")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    target_precession = _finite(target_precession_deg_day, "target_precession_deg_day")
    j2_value = _finite(j2, "j2")
    if e >= 1.0:
        raise ValueError("eccentricity must be less than 1 for an elliptical orbit.")
    a = body_radius + altitude
    n = sqrt(mu / (a**3))
    p = a * (1.0 - e * e)
    target_rad_s = radians(target_precession) / 86400.0
    denominator = 1.5 * j2_value * n * (body_radius / p) ** 2
    cos_i = -target_rad_s / denominator
    if cos_i < -1.0 or cos_i > 1.0:
        raise ValueError("No first-order circular sun-synchronous inclination exists for this altitude.")
    inclination = degrees(acos(cos_i))
    actual = -1.5 * j2_value * n * (body_radius / p) ** 2 * cos(radians(inclination))
    return SunSynchronousInclinationResult(
        altitude_km=altitude,
        semi_major_axis_km=a,
        inclination_deg=inclination,
        nodal_precession_deg_day=degrees(actual) * 86400.0,
        target_precession_deg_day=target_precession,
    )


def j2_secular_rates_from_elements(
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
    j2: float = EARTH_J2,
) -> J2SecularRatesResult:
    a = _positive(semi_major_axis_km, "semi_major_axis_km")
    e = _nonnegative(eccentricity, "eccentricity")
    body_radius = _positive(body_radius_km, "body_radius_km")
    j2_value = _finite(j2, "j2")
    if e >= 1.0:
        raise ValueError("eccentricity must be less than 1 for an elliptical orbit.")
    inclination = _finite(inclination_deg, "inclination_deg")
    if inclination < 0.0 or inclination > 180.0:
        raise ValueError("inclination_deg must be between 0 and 180.")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    p = a * (1.0 - e * e)
    n = sqrt(mu / (a**3))
    cos_i = cos(radians(inclination))
    factor = j2_value * n * (body_radius / p) ** 2
    raan_rate_rad_s = -1.5 * factor * cos_i
    arg_perigee_rate_rad_s = 0.75 * factor * ((5.0 * cos_i * cos_i) - 1.0)
    mean_anomaly_rate_rad_s = 0.75 * factor * sqrt(1.0 - e * e) * ((3.0 * cos_i * cos_i) - 1.0)
    scale = 86400.0 * 180.0 / pi
    return J2SecularRatesResult(
        semi_major_axis_km=a,
        eccentricity=e,
        inclination_deg=inclination,
        semi_latus_rectum_km=p,
        mean_motion_rad_s=n,
        orbital_period_s=2.0 * pi / n,
        raan_rate_deg_day=raan_rate_rad_s * scale,
        argument_of_perigee_rate_deg_day=arg_perigee_rate_rad_s * scale,
        mean_anomaly_j2_rate_deg_day=mean_anomaly_rate_rad_s * scale,
    )


def j2_secular_rates_from_altitude(
    altitude_km: float,
    eccentricity: float,
    inclination_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
    j2: float = EARTH_J2,
) -> J2SecularRatesResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    body_radius = _positive(body_radius_km, "body_radius_km")
    return j2_secular_rates_from_elements(
        semi_major_axis_km=body_radius + altitude,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        mu_km3_s2=mu_km3_s2,
        body_radius_km=body_radius_km,
        j2=j2,
    )


def hcw_natural_motion_from_altitude(
    altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> HCWNaturalMotionResult:
    circular = circular_orbit_from_altitude(altitude_km, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    return HCWNaturalMotionResult(
        altitude_km=circular.altitude_km,
        orbit_radius_km=circular.radius_km,
        mean_motion_rad_s=circular.mean_motion_rad_s,
        natural_motion_period_s=circular.period_s,
        circular_velocity_km_s=circular.velocity_km_s,
    )


def hcw_intrack_drift_estimate(
    altitude_km: float,
    radial_offset_m: float,
    intrack_velocity_bias_m_s: float,
    duration_orbits: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> HCWDriftEstimateResult:
    natural = hcw_natural_motion_from_altitude(altitude_km, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    duration = _nonnegative(duration_orbits, "duration_orbits") * natural.natural_motion_period_s
    nt = natural.mean_motion_rad_s * duration
    radial_offset_km = float(radial_offset_m) / 1000.0
    intrack_velocity_bias_km_s = float(intrack_velocity_bias_m_s) / 1000.0
    radial_offset_drift_km = -6.0 * radial_offset_km * (nt - sin(nt))
    intrack_velocity_drift_km = ((4.0 * intrack_velocity_bias_km_s / natural.mean_motion_rad_s) * sin(nt)) - (
        3.0 * intrack_velocity_bias_km_s * duration
    )
    total_drift_km = radial_offset_drift_km + intrack_velocity_drift_km
    return HCWDriftEstimateResult(
        altitude_km=natural.altitude_km,
        duration_orbits=float(duration_orbits),
        duration_s=duration,
        radial_offset_m=float(radial_offset_m),
        intrack_velocity_bias_m_s=float(intrack_velocity_bias_m_s),
        radial_offset_drift_m=radial_offset_drift_km * 1000.0,
        intrack_velocity_drift_m=intrack_velocity_drift_km * 1000.0,
        total_intrack_drift_m=total_drift_km * 1000.0,
        mean_motion_rad_s=natural.mean_motion_rad_s,
    )


def circular_eclipse_estimate(
    altitude_km: float,
    beta_angle_deg: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> EclipseEstimateResult:
    circular = circular_orbit_from_altitude(altitude_km, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    beta_abs = abs(float(beta_angle_deg))
    if beta_abs > 90.0:
        raise ValueError("beta_angle_deg magnitude must be no greater than 90.")
    beta_critical = degrees(asin(min(1.0, float(body_radius_km) / circular.radius_km)))
    eclipse_half_angle = None
    eclipse_duration = 0.0
    eclipse_fraction = 0.0
    if beta_abs < beta_critical:
        denominator = max(cos(radians(beta_abs)), 1.0e-15)
        argument = sqrt(max(0.0, 1.0 - (float(body_radius_km) / circular.radius_km) ** 2)) / denominator
        eclipse_half_angle_rad = acos(float(np.clip(argument, -1.0, 1.0)))
        eclipse_half_angle = degrees(eclipse_half_angle_rad)
        eclipse_fraction = eclipse_half_angle_rad / pi
        eclipse_duration = eclipse_fraction * circular.period_s
    return EclipseEstimateResult(
        altitude_km=circular.altitude_km,
        beta_angle_deg=float(beta_angle_deg),
        orbit_radius_km=circular.radius_km,
        orbital_period_s=circular.period_s,
        beta_critical_deg=beta_critical,
        eclipse_half_angle_deg=eclipse_half_angle,
        eclipse_duration_s=eclipse_duration,
        eclipse_fraction=eclipse_fraction,
        sunlight_duration_s=circular.period_s - eclipse_duration,
    )


def ground_track_drift_from_altitude(
    altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> GroundTrackDriftResult:
    circular = circular_orbit_from_altitude(altitude_km, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    earth_rotation_deg = degrees(EARTH_ROTATION_RATE_RAD_S * circular.period_s)
    drift_deg = -earth_rotation_deg
    drift_km = (pi / 180.0) * float(body_radius_km) * drift_deg
    return GroundTrackDriftResult(
        altitude_km=circular.altitude_km,
        orbital_period_s=circular.period_s,
        earth_rotation_deg_per_orbit=earth_rotation_deg,
        westward_drift_deg_per_orbit=drift_deg,
        orbits_per_sidereal_day=SIDEREAL_DAY_S / circular.period_s,
        equator_drift_km_per_orbit=drift_km,
    )


def repeat_ground_track_approximation(
    altitude_km: float,
    max_days: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> RepeatGroundTrackResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    max_repeat_days = _positive_int(max_days, "max_days")
    circular = circular_orbit_from_altitude(altitude, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    best: tuple[float, int, int, float] | None = None
    for days in range(1, max_repeat_days + 1):
        ideal_orbits = days * SIDEREAL_DAY_S / circular.period_s
        orbits = max(1, int(round(ideal_orbits)))
        period_error_s = (days * SIDEREAL_DAY_S / orbits) - circular.period_s
        longitude_error_deg = 360.0 * period_error_s / circular.period_s
        cycle_error_deg = longitude_error_deg * orbits
        abs_error = abs(cycle_error_deg)
        if best is None or abs_error < best[0]:
            best = (abs_error, days, orbits, longitude_error_deg)
    if best is None:
        raise ValueError("max_days must include at least one day.")
    _, days, orbits, longitude_error_deg_per_orbit = best
    longitude_error_deg = longitude_error_deg_per_orbit * orbits
    exact_period = days * SIDEREAL_DAY_S / orbits
    exact_radius = (float(mu_km3_s2) / ((2.0 * pi / exact_period) ** 2)) ** (1.0 / 3.0)
    return RepeatGroundTrackResult(
        altitude_km=altitude,
        max_days=max_repeat_days,
        repeat_days=days,
        repeat_orbits=orbits,
        ground_track_error_deg=longitude_error_deg,
        ground_track_error_km=(pi / 180.0) * float(body_radius_km) * longitude_error_deg,
        exact_repeat_altitude_km=exact_radius - float(body_radius_km),
        exact_repeat_period_s=exact_period,
    )


def entry_interface_estimate(
    apogee_altitude_km: float,
    perigee_altitude_km: float,
    interface_altitude_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> EntryInterfaceEstimateResult:
    elements = elements_from_apogee_perigee_altitudes(
        perigee_altitude_km=perigee_altitude_km,
        apogee_altitude_km=apogee_altitude_km,
        body_radius_km=body_radius_km,
    )
    interface_altitude = _nonnegative(interface_altitude_km, "interface_altitude_km")
    interface_radius = float(body_radius_km) + interface_altitude
    if interface_radius < elements.perigee_radius_km or interface_radius > elements.apogee_radius_km:
        raise ValueError("interface_altitude_km must lie between perigee_altitude_km and apogee_altitude_km.")
    speed = vis_viva_velocity(interface_radius, elements.semi_major_axis_km, mu_km3_s2=mu_km3_s2).velocity_km_s
    if elements.eccentricity <= ECCENTRICITY_SINGULAR_TOL:
        true_anomaly = 0.0
        flight_path_angle = 0.0
    else:
        p = elements.semi_major_axis_km * (1.0 - elements.eccentricity * elements.eccentricity)
        cos_nu = float(np.clip(((p / interface_radius) - 1.0) / elements.eccentricity, -1.0, 1.0))
        true_anomaly = 360.0 - degrees(acos(cos_nu))
        sin_nu = -sqrt(max(0.0, 1.0 - cos_nu * cos_nu))
        flight_path_angle = degrees(np.arctan2(elements.eccentricity * sin_nu, 1.0 + elements.eccentricity * cos_nu))
    abs_fpa = abs(flight_path_angle)
    if abs_fpa < 1.0:
        note = "Very shallow vacuum interface angle; atmospheric and lift/drag effects will dominate real entry behavior."
    elif abs_fpa <= 5.0:
        note = "Moderate vacuum interface angle; still not a heating, loads, or survivability estimate."
    else:
        note = "Steep vacuum interface angle; treat as geometry intuition only, not an entry safety estimate."
    return EntryInterfaceEstimateResult(
        apogee_altitude_km=elements.apogee_altitude_km,
        perigee_altitude_km=elements.perigee_altitude_km,
        interface_altitude_km=interface_altitude,
        semi_major_axis_km=elements.semi_major_axis_km,
        eccentricity=elements.eccentricity,
        interface_radius_km=interface_radius,
        speed_km_s=speed,
        flight_path_angle_deg=flight_path_angle,
        true_anomaly_deg=true_anomaly,
        note=note,
    )


def phasing_drift_from_altitude_change(
    reference_altitude_km: float,
    phasing_altitude_change_km: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> PhasingDriftResult:
    reference_altitude = _nonnegative(reference_altitude_km, "reference_altitude_km")
    altitude_change = float(phasing_altitude_change_km)
    phasing_altitude = reference_altitude + altitude_change
    if phasing_altitude < 0.0:
        raise ValueError("phasing altitude must be non-negative.")
    ref = circular_orbit_from_altitude(reference_altitude, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    phasing = circular_orbit_from_altitude(phasing_altitude, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    delta_n = phasing.mean_motion_rad_s - ref.mean_motion_rad_s
    drift_rate_km_s = ref.radius_km * delta_n
    lap_time_s = None if abs(delta_n) < 1.0e-15 else (2.0 * pi / abs(delta_n))
    return PhasingDriftResult(
        reference_altitude_km=reference_altitude,
        phasing_altitude_km=phasing_altitude,
        altitude_change_km=altitude_change,
        reference_mean_motion_rad_s=ref.mean_motion_rad_s,
        phasing_mean_motion_rad_s=phasing.mean_motion_rad_s,
        drift_rate_m_s=drift_rate_km_s * 1000.0,
        drift_per_reference_orbit_km=drift_rate_km_s * ref.period_s,
        lap_time_s=lap_time_s,
    )


def mission_recovery_from_intrack_impulse(
    reference_altitude_km: float,
    disturbance_delta_v_m_s: float,
    spacecraft_mass_kg: float,
    isp_s: float,
    *,
    slot_tolerance_deg: float = 1.0,
    max_phasing_orbits: int = 5000,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
    standard_gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
) -> MissionRecoveryIntrackImpulseResult:
    altitude = _nonnegative(reference_altitude_km, "reference_altitude_km")
    disturbance = float(disturbance_delta_v_m_s)
    if not np.isfinite(disturbance):
        raise ValueError("disturbance_delta_v_m_s must be finite.")
    mass = _positive(spacecraft_mass_kg, "spacecraft_mass_kg")
    isp = _positive(isp_s, "isp_s")
    tolerance = _nonnegative(slot_tolerance_deg, "slot_tolerance_deg")
    max_orbits = _positive_int(max_phasing_orbits, "max_phasing_orbits")

    reference = circular_orbit_from_altitude(altitude, mu_km3_s2=mu_km3_s2, body_radius_km=body_radius_km)
    r0 = float(reference.radius_km)
    v_circ = float(reference.velocity_km_s)
    disturbed_speed = v_circ + disturbance / 1000.0
    if disturbed_speed <= 0.0:
        raise ValueError("disturbance_delta_v_m_s makes the disturbed orbital speed non-positive.")

    energy = 0.5 * disturbed_speed * disturbed_speed - float(mu_km3_s2) / r0
    if energy >= -ENERGY_PARABOLIC_TOL:
        raise ValueError("disturbance_delta_v_m_s creates a parabolic or escaping trajectory.")
    disturbed_a = -float(mu_km3_s2) / (2.0 * energy)
    if disturbed_a <= 0.0:
        raise ValueError("disturbed semi-major axis must be positive.")
    if disturbance < 0.0:
        disturbance_apsis = "apogee"
        apogee_radius = r0
        perigee_radius = 2.0 * disturbed_a - apogee_radius
    elif disturbance > 0.0:
        disturbance_apsis = "perigee"
        perigee_radius = r0
        apogee_radius = 2.0 * disturbed_a - perigee_radius
    else:
        disturbance_apsis = "circular"
        perigee_radius = r0
        apogee_radius = r0
    if perigee_radius <= 0.0 or apogee_radius <= 0.0:
        raise ValueError("disturbance_delta_v_m_s creates an invalid elliptical orbit.")
    disturbed_ecc = (apogee_radius - perigee_radius) / (apogee_radius + perigee_radius)
    disturbed_period = 2.0 * pi * sqrt((disturbed_a**3) / float(mu_km3_s2))

    recovery_dv = abs(disturbance)
    recovery_mass = rocket_equation_mass_ratio(
        delta_v_m_s=recovery_dv,
        isp_s=isp,
        standard_gravity_m_s2=standard_gravity_m_s2,
    )
    total_event_mass = rocket_equation_mass_ratio(
        delta_v_m_s=2.0 * recovery_dv,
        isp_s=isp,
        standard_gravity_m_s2=standard_gravity_m_s2,
    )
    recovery_propellant_kg = mass * recovery_mass.propellant_fraction

    n_ref = float(reference.mean_motion_rad_s)
    n_disturbed = sqrt(float(mu_km3_s2) / (disturbed_a**3))
    delta_n = n_disturbed - n_ref
    continuous_lap = None if abs(delta_n) < 1.0e-15 else 2.0 * pi / abs(delta_n)

    best_orbits = 1
    best_error = float("inf")
    found_orbits: int | None = None
    found_error: float | None = None
    for orbit_count in range(1, max_orbits + 1):
        reference_phase = degrees(n_ref * disturbed_period * orbit_count)
        phase_error = abs(_wrap_signed_degrees(reference_phase))
        if phase_error < best_error:
            best_error = phase_error
            best_orbits = orbit_count
        if found_orbits is None and phase_error <= tolerance:
            found_orbits = orbit_count
            found_error = phase_error
            break

    notes: list[str] = [
        "Impulsive two-body estimate from an initially circular reference orbit.",
        "Recovery burn is the opposite in-track impulse applied at the same apsis of the disturbed phasing orbit.",
    ]
    perigee_altitude = perigee_radius - float(body_radius_km)
    apogee_altitude = apogee_radius - float(body_radius_km)
    if perigee_altitude < 0.0:
        notes.append("Disturbed orbit intersects the central body before a full phasing orbit.")
    if found_orbits is None:
        notes.append("No discrete same-apsis slot recovery found within the requested tolerance and search window.")
    elif found_orbits == 1:
        notes.append("First same-apsis recovery opportunity is inside the requested slot tolerance.")

    return MissionRecoveryIntrackImpulseResult(
        reference_altitude_km=altitude,
        reference_radius_km=r0,
        reference_period_s=float(reference.period_s),
        circular_speed_km_s=v_circ,
        disturbance_delta_v_m_s=disturbance,
        disturbed_speed_km_s=float(disturbed_speed),
        disturbed_semi_major_axis_km=float(disturbed_a),
        disturbed_eccentricity=float(disturbed_ecc),
        disturbed_perigee_altitude_km=float(perigee_altitude),
        disturbed_apogee_altitude_km=float(apogee_altitude),
        disturbed_period_s=float(disturbed_period),
        disturbance_apsis=disturbance_apsis,
        recovery_delta_v_m_s=float(recovery_dv),
        recovery_propellant_kg=float(recovery_propellant_kg),
        recovery_propellant_fraction=float(recovery_mass.propellant_fraction),
        total_event_delta_v_m_s=float(2.0 * recovery_dv),
        total_event_propellant_fraction=float(total_event_mass.propellant_fraction),
        spacecraft_mass_kg=mass,
        isp_s=isp,
        slot_tolerance_deg=float(tolerance),
        max_phasing_orbits=max_orbits,
        continuous_slot_lap_time_s=continuous_lap,
        slot_recovery_found=found_orbits is not None,
        slot_recovery_orbits=found_orbits,
        slot_recovery_time_s=(None if found_orbits is None else float(found_orbits * disturbed_period)),
        slot_recovery_phase_error_deg=found_error,
        best_slot_orbits=int(best_orbits),
        best_slot_time_s=float(best_orbits * disturbed_period),
        best_slot_phase_error_deg=float(best_error),
        notes=tuple(notes),
    )


def rocket_equation_delta_v(
    isp_s: float,
    mass_ratio: float,
    *,
    standard_gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
) -> RocketEquationDeltaVResult:
    isp = _positive(isp_s, "isp_s")
    ratio = _positive(mass_ratio, "mass_ratio")
    standard_gravity = _positive(standard_gravity_m_s2, "standard_gravity_m_s2")
    if ratio < 1.0:
        raise ValueError("mass_ratio must be greater than or equal to 1.")
    delta_v = isp * standard_gravity * log(ratio)
    return RocketEquationDeltaVResult(isp_s=isp, mass_ratio=ratio, delta_v_m_s=delta_v)


def rocket_equation_mass_ratio(
    delta_v_m_s: float,
    isp_s: float,
    *,
    standard_gravity_m_s2: float = STANDARD_GRAVITY_M_S2,
) -> RocketEquationMassRatioResult:
    delta_v = _nonnegative(delta_v_m_s, "delta_v_m_s")
    isp = _positive(isp_s, "isp_s")
    standard_gravity = _positive(standard_gravity_m_s2, "standard_gravity_m_s2")
    ratio = exp(delta_v / (isp * standard_gravity))
    return RocketEquationMassRatioResult(
        isp_s=isp,
        delta_v_m_s=delta_v,
        mass_ratio=ratio,
        propellant_fraction=1.0 - (1.0 / ratio),
    )


def atmospheric_lifetime_warning(altitude_km: float) -> str:
    altitude = float(altitude_km)
    if altitude > USSA1976_MAX_ALTITUDE_KM:
        return (
            "Outside built-in density-table range: USSA-1976 calculator returns zero density above "
            f"{USSA1976_MAX_ALTITUDE_KM:.0f} km."
        )
    if altitude < 250.0:
        return "Very low orbit: decay can be rapid without propulsion."
    if altitude < 400.0:
        return "Drag-dominated LEO: lifetime is highly sensitive to area, mass, and space weather."
    if altitude < 600.0:
        return "Meaningful drag: decay can matter over mission timescales depending on ballistic coefficient."
    if altitude < 800.0:
        return "Light but nonzero drag: useful for napkin estimates, still space-weather sensitive."
    return "Very low modeled density: drag is often small for first-pass circular-orbit estimates."


def atmospheric_density_from_altitude(altitude_km: float) -> AtmosphericDensityResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    r_eci_km = np.array([EARTH_RADIUS_KM + altitude, 0.0, 0.0], dtype=float)
    return AtmosphericDensityResult(
        altitude_km=altitude,
        density_kg_m3=density_ussa1976(r_eci_km, 0.0),
        warning=atmospheric_lifetime_warning(altitude),
    )


def ballistic_coefficient(
    mass_kg: float,
    drag_coefficient: float,
    drag_area_m2: float,
) -> BallisticCoefficientResult:
    mass = _positive(mass_kg, "mass_kg")
    cd = _positive(drag_coefficient, "drag_coefficient")
    area = _positive(drag_area_m2, "drag_area_m2")
    return BallisticCoefficientResult(
        mass_kg=mass,
        drag_coefficient=cd,
        drag_area_m2=area,
        ballistic_coefficient_kg_m2=mass / (cd * area),
    )


def drag_force_acceleration_from_density_speed(
    density_kg_m3: float,
    speed_m_s: float,
    drag_coefficient: float,
    drag_area_m2: float,
    mass_kg: float,
) -> DragForceResult:
    density = _nonnegative(density_kg_m3, "density_kg_m3")
    speed = _nonnegative(speed_m_s, "speed_m_s")
    bc = ballistic_coefficient(mass_kg, drag_coefficient, drag_area_m2)
    force = 0.5 * density * speed * speed * bc.drag_coefficient * bc.drag_area_m2
    return DragForceResult(
        density_kg_m3=density,
        speed_m_s=speed,
        drag_coefficient=bc.drag_coefficient,
        drag_area_m2=bc.drag_area_m2,
        mass_kg=bc.mass_kg,
        drag_force_n=force,
        drag_accel_m_s2=force / bc.mass_kg,
        ballistic_coefficient_kg_m2=bc.ballistic_coefficient_kg_m2,
    )


def drag_force_acceleration_from_altitude(
    altitude_km: float,
    mass_kg: float,
    drag_coefficient: float,
    drag_area_m2: float,
) -> DragAtAltitudeResult:
    density = atmospheric_density_from_altitude(altitude_km)
    circular = circular_orbit_from_altitude(altitude_km)
    drag = drag_force_acceleration_from_density_speed(
        density_kg_m3=density.density_kg_m3,
        speed_m_s=circular.velocity_km_s * 1000.0,
        drag_coefficient=drag_coefficient,
        drag_area_m2=drag_area_m2,
        mass_kg=mass_kg,
    )
    return DragAtAltitudeResult(
        altitude_km=density.altitude_km,
        density_kg_m3=density.density_kg_m3,
        circular_speed_m_s=drag.speed_m_s,
        drag_coefficient=drag.drag_coefficient,
        drag_area_m2=drag.drag_area_m2,
        mass_kg=drag.mass_kg,
        drag_force_n=drag.drag_force_n,
        drag_accel_m_s2=drag.drag_accel_m_s2,
        ballistic_coefficient_kg_m2=drag.ballistic_coefficient_kg_m2,
        warning=density.warning,
    )


def circular_orbit_drag_decay_rate(
    altitude_km: float,
    ballistic_coefficient_kg_m2: float,
    *,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> CircularOrbitDecayRateResult:
    altitude = _nonnegative(altitude_km, "altitude_km")
    if altitude > USSA1976_MAX_ALTITUDE_KM:
        raise ValueError(
            "altitude_km is above the 1000 km limit of the built-in USSA-1976 density table. "
            "Use a higher-fidelity propagator or external density model."
        )
    bc = _positive(ballistic_coefficient_kg_m2, "ballistic_coefficient_kg_m2")
    mu = _positive(mu_km3_s2, "mu_km3_s2")
    body_radius = _positive(body_radius_km, "body_radius_km")
    density = atmospheric_density_from_altitude(altitude)
    mu_m3_s2 = mu * 1.0e9
    semi_major_axis_m = (body_radius + altitude) * 1000.0
    decay_rate_m_s = -(density.density_kg_m3 / bc) * sqrt(mu_m3_s2 * semi_major_axis_m)
    decay_rate_m_day = decay_rate_m_s * 86400.0
    return CircularOrbitDecayRateResult(
        altitude_km=altitude,
        density_kg_m3=density.density_kg_m3,
        ballistic_coefficient_kg_m2=bc,
        decay_rate_m_s=decay_rate_m_s,
        decay_rate_m_day=decay_rate_m_day,
        decay_rate_km_day=decay_rate_m_day / 1000.0,
        warning=density.warning,
    )


def _drag_lifetime_for_density_scale(
    *,
    initial_altitude_km: float,
    deorbit_altitude_km: float,
    ballistic_coefficient_kg_m2: float,
    density_scale: float,
    integration_step_km: float,
    mu_km3_s2: float,
    body_radius_km: float,
) -> float:
    scale = _positive(density_scale, "density_scale")
    step = _positive(integration_step_km, "integration_step_km")
    altitude = float(initial_altitude_km)
    total_s = 0.0
    while altitude > deorbit_altitude_km:
        next_altitude = max(float(deorbit_altitude_km), altitude - step)
        mid_altitude = 0.5 * (altitude + next_altitude)
        local = circular_orbit_drag_decay_rate(
            mid_altitude,
            ballistic_coefficient_kg_m2,
            mu_km3_s2=mu_km3_s2,
            body_radius_km=body_radius_km,
        )
        local_decay_m_s = abs(local.decay_rate_m_s) * scale
        if local_decay_m_s <= 0.0:
            return float("inf")
        total_s += ((altitude - next_altitude) * 1000.0) / local_decay_m_s
        altitude = next_altitude
    return total_s


def drag_lifetime_range_estimate(
    initial_altitude_km: float,
    ballistic_coefficient_kg_m2: float,
    deorbit_altitude_km: float,
    *,
    low_drag_density_scale: float = 0.3,
    nominal_density_scale: float = 1.0,
    high_drag_density_scale: float = 3.0,
    integration_step_km: float = 1.0,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
    body_radius_km: float = EARTH_RADIUS_KM,
) -> DragLifetimeRangeResult:
    initial_altitude = _nonnegative(initial_altitude_km, "initial_altitude_km")
    deorbit_altitude = _nonnegative(deorbit_altitude_km, "deorbit_altitude_km")
    bc = _positive(ballistic_coefficient_kg_m2, "ballistic_coefficient_kg_m2")
    if initial_altitude > USSA1976_MAX_ALTITUDE_KM:
        raise ValueError(
            "initial_altitude_km is above the 1000 km limit of the built-in USSA-1976 density table. "
            "Use a higher-fidelity propagator or external density model."
        )
    if deorbit_altitude >= initial_altitude:
        raise ValueError("deorbit_altitude_km must be lower than initial_altitude_km.")
    step = _positive(integration_step_km, "integration_step_km")

    common = {
        "initial_altitude_km": initial_altitude,
        "deorbit_altitude_km": deorbit_altitude,
        "ballistic_coefficient_kg_m2": bc,
        "integration_step_km": step,
        "mu_km3_s2": mu_km3_s2,
        "body_radius_km": body_radius_km,
    }
    low_scale = _positive(low_drag_density_scale, "low_drag_density_scale")
    nominal_scale = _positive(nominal_density_scale, "nominal_density_scale")
    high_scale = _positive(high_drag_density_scale, "high_drag_density_scale")
    return DragLifetimeRangeResult(
        initial_altitude_km=initial_altitude,
        deorbit_altitude_km=deorbit_altitude,
        ballistic_coefficient_kg_m2=bc,
        low_drag_density_scale=low_scale,
        nominal_density_scale=nominal_scale,
        high_drag_density_scale=high_scale,
        low_drag_lifetime_s=_drag_lifetime_for_density_scale(density_scale=low_scale, **common),
        nominal_lifetime_s=_drag_lifetime_for_density_scale(density_scale=nominal_scale, **common),
        high_drag_lifetime_s=_drag_lifetime_for_density_scale(density_scale=high_scale, **common),
        integration_step_km=step,
        warning=atmospheric_lifetime_warning(initial_altitude),
    )
