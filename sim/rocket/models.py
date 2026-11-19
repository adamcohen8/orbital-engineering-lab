from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sim.numeric_backend import normalize_numeric_backend
from sim.presets.rockets import RocketStackPreset
from sim.rocket.aero import RocketAeroConfig
from sim.rocket.landing_gear import LandingGear, LandingGearCommand, LegState
from sim.rocket.touchdown import RocketTouchdownLimits


@dataclass(frozen=True)
class RocketSimConfig:
    dt_s: float = 0.5
    max_time_s: float = 2000.0
    target_altitude_km: float = 400.0
    target_altitude_tolerance_km: float = 25.0
    target_eccentricity_max: float = 0.02
    insertion_hold_time_s: float = 30.0
    launch_lat_deg: float = 0.0
    launch_lon_deg: float = 0.0
    launch_alt_km: float = 0.0
    launch_azimuth_deg: float = 90.0
    atmosphere_model: str = "ussa1976"
    enable_drag: bool = True
    enable_srp: bool = False
    enable_j2: bool = True
    enable_j3: bool = False
    enable_j4: bool = False
    terminate_on_earth_impact: bool = True
    earth_impact_radius_km: float = 6378.137
    area_ref_m2: float | None = None
    use_stagewise_aero_geometry: bool = True
    cd: float = 0.35
    cr: float = 1.2
    aero: RocketAeroConfig = field(default_factory=RocketAeroConfig)
    atmosphere_env: dict = field(default_factory=dict)
    use_wgs84_geodesy: bool = True
    wind_enu_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_enu_callable: object | None = None
    inertia_kg_m2: np.ndarray = field(default_factory=lambda: np.diag([8.0e5, 8.0e5, 2.0e4]))
    attitude_substep_s: float = 0.02
    attitude_mode: str = "dynamic"  # dynamic | cheater
    tvc_time_constant_s: float = 0.1
    tvc_max_gimbal_deg: float = 6.0
    tvc_rate_limit_deg_s: float = 20.0
    tvc_pivot_offset_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    numeric_backend: str = "rust"
    touchdown: RocketTouchdownLimits | None = None

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self.max_time_s <= 0.0:
            raise ValueError("max_time_s must be positive.")
        if self.target_altitude_tolerance_km < 0.0:
            raise ValueError("target_altitude_tolerance_km must be non-negative.")
        if self.target_eccentricity_max < 0.0:
            raise ValueError("target_eccentricity_max must be non-negative.")
        if self.insertion_hold_time_s < 0.0:
            raise ValueError("insertion_hold_time_s must be non-negative.")
        if self.attitude_substep_s <= 0.0:
            raise ValueError("attitude_substep_s must be positive.")
        if self.earth_impact_radius_km <= 0.0:
            raise ValueError("earth_impact_radius_km must be positive.")
        if self.touchdown is not None and not self.terminate_on_earth_impact:
            raise ValueError("Rocket touchdown requires terminate_on_earth_impact.")
        if np.array(self.wind_enu_m_s, dtype=float).reshape(-1).size != 3:
            raise ValueError("wind_enu_m_s must be length-3.")
        if self.tvc_time_constant_s <= 0.0:
            raise ValueError("tvc_time_constant_s must be positive.")
        if self.tvc_max_gimbal_deg < 0.0:
            raise ValueError("tvc_max_gimbal_deg must be non-negative.")
        if self.tvc_rate_limit_deg_s < 0.0:
            raise ValueError("tvc_rate_limit_deg_s must be non-negative.")
        if np.array(self.tvc_pivot_offset_body_m, dtype=float).reshape(-1).size != 3:
            raise ValueError("tvc_pivot_offset_body_m must be length-3.")
        mode = str(self.attitude_mode).strip().lower()
        if mode not in ("dynamic", "cheater"):
            raise ValueError("attitude_mode must be 'dynamic' or 'cheater'.")
        backend = normalize_numeric_backend(self.numeric_backend, error_message="numeric_backend must be 'python' or 'rust'.")
        object.__setattr__(self, "numeric_backend", backend)


@dataclass(frozen=True)
class RocketVehicleConfig:
    stack: RocketStackPreset
    payload_mass_kg: float
    thrust_axis_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    landing_gear: LandingGear | None = None

    def __post_init__(self) -> None:
        if self.payload_mass_kg < 0.0:
            raise ValueError("payload_mass_kg must be non-negative.")
        axis = np.array(self.thrust_axis_body, dtype=float).reshape(-1)
        if axis.size != 3:
            raise ValueError("thrust_axis_body must be length-3.")
        if np.linalg.norm(axis) <= 0.0:
            raise ValueError("thrust_axis_body cannot be zero.")


@dataclass
class RocketState:
    t_s: float
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: float
    active_stage_index: int
    stage_prop_remaining_kg: np.ndarray
    payload_attached: bool = True
    thrust_vector_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float))
    landing_legs: dict[str, LegState] = field(default_factory=dict)
    gear_touchdown: dict | None = None
    gear_terminal: dict | None = None
    gear_settle_elapsed_s: float = 0.0
    gear_rebound_time_s: float | None = None

    def copy(self) -> RocketState:
        return RocketState(
            t_s=float(self.t_s),
            position_eci_km=self.position_eci_km.copy(),
            velocity_eci_km_s=self.velocity_eci_km_s.copy(),
            attitude_quat_bn=self.attitude_quat_bn.copy(),
            angular_rate_body_rad_s=self.angular_rate_body_rad_s.copy(),
            mass_kg=float(self.mass_kg),
            active_stage_index=int(self.active_stage_index),
            stage_prop_remaining_kg=self.stage_prop_remaining_kg.copy(),
            payload_attached=bool(self.payload_attached),
            thrust_vector_body=self.thrust_vector_body.copy(),
            landing_legs=self.landing_legs.copy(),
            gear_touchdown=deepcopy(self.gear_touchdown),
            gear_terminal=deepcopy(self.gear_terminal),
            gear_settle_elapsed_s=self.gear_settle_elapsed_s,
            gear_rebound_time_s=self.gear_rebound_time_s,
        )


@dataclass(frozen=True)
class GuidanceCommand:
    throttle: float
    attitude_quat_bn_cmd: np.ndarray | None = None
    torque_body_nm_cmd: np.ndarray | None = None
    thrust_vector_body_cmd: np.ndarray | None = None
    landing_gear_commands: tuple[LandingGearCommand, ...] = ()


class RocketGuidanceLaw(Protocol):
    def command(
        self, state: RocketState, sim_cfg: RocketSimConfig, vehicle_cfg: RocketVehicleConfig
    ) -> GuidanceCommand: ...


@dataclass
class RocketSimResult:
    time_s: np.ndarray
    position_eci_km: np.ndarray
    velocity_eci_km_s: np.ndarray
    attitude_quat_bn: np.ndarray
    angular_rate_body_rad_s: np.ndarray
    mass_kg: np.ndarray
    active_stage_index: np.ndarray
    throttle_cmd: np.ndarray
    thrust_n: np.ndarray
    altitude_km: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    eccentricity: np.ndarray
    sma_km: np.ndarray
    dynamic_pressure_pa: np.ndarray
    mach: np.ndarray
    wind_body_m_s: np.ndarray
    tvc_gimbal_deg: np.ndarray
    alpha_deg: np.ndarray
    beta_deg: np.ndarray
    cd: np.ndarray
    aero_force_n: np.ndarray
    aero_moment_nm: np.ndarray
    inserted: bool
    insertion_time_s: float | None
    terminated_early: bool = False
    termination_reason: str | None = None
    termination_time_s: float | None = None
    touchdown: dict | None = None
    landing_gear: dict | None = None
