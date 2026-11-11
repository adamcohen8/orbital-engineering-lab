from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ControllerBenchTarget:
    object_id: str
    slot: str


@dataclass(frozen=True)
class ControllerBenchOptimizationParameter:
    name: str
    path: str
    lower: float | tuple[Any, ...]
    upper: float | tuple[Any, ...]


@dataclass(frozen=True)
class ControllerBenchOptimizationCost:
    object_id: str
    reference_object_id: str
    terminal_state_weight: float = 1.0
    terminal_position_scale_km: float = 0.01
    terminal_speed_scale_km_s: float = 0.001
    fuel_weight: float = 1.0
    fuel_scale_kg: float = 1.0
    time_weight: float = 1.0
    rendezvous_radius_km: float = 0.01
    time_scale_s: float = 600.0
    miss_time_penalty_s: float = 3600.0


@dataclass(frozen=True)
class ControllerBenchOptimizationConfig:
    enabled: bool = False
    algorithm: str = "pso"
    base_variant: str = ""
    particles: int = 8
    iterations: int = 10
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    seed: int = 0
    parameters: tuple[ControllerBenchOptimizationParameter, ...] = ()
    cost: ControllerBenchOptimizationCost | None = None


@dataclass(frozen=True)
class ControllerBenchEnvelopeVariation:
    name: str
    path: str
    values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ControllerBenchEnvelopeConfig:
    enabled: bool = False
    seed: int = 0
    max_cases: int = 0
    base_cases: tuple[str, ...] = ()
    variations: tuple[ControllerBenchEnvelopeVariation, ...] = ()


@dataclass(frozen=True)
class ControllerVariantPointer:
    object_id: str
    slot: str
    pointer: dict[str, Any]


@dataclass(frozen=True)
class ControllerVariantOverride:
    path: str
    value: Any


@dataclass(frozen=True)
class ControllerVariant:
    name: str
    pointer: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    pointers: tuple[ControllerVariantPointer, ...] = ()
    overrides: tuple[ControllerVariantOverride, ...] = ()


@dataclass(frozen=True)
class ControllerBenchMetric:
    name: str
    source_path: str = ""
    kind: str = ""
    object_id: str = ""
    reference_object_id: str = ""
    desired_quat_bn: tuple[float, float, float, float] | None = None
    keepout_radius_km: float | None = None
    settle_error_deg: float | None = None
    settle_rate_norm_rad_s: float | None = None
    settle_hold_s: float | None = None
    target_relative_state_ric_m: tuple[float, float, float, float, float, float] | None = None
    target_value: float | None = None
    expected_value: str = ""
    diagnostic_field: str = ""
    waypoints_ric_m: tuple[tuple[float, float, float, float, float, float], ...] = ()
    waypoint_tolerance_m: float | None = None
    waypoint_velocity_tolerance_m_s: float | None = None


@dataclass(frozen=True)
class ControllerBenchPassCriterion:
    metric: str
    op: str
    value: Any


@dataclass(frozen=True)
class ControllerBenchObjective:
    kind: str
    name: str = ""
    object_id: str = ""
    reference_object_id: str = ""
    desired_quat_bn: tuple[float, float, float, float] | None = None
    keepout_radius_km: float | None = None
    settle_error_deg: float | None = None
    settle_rate_norm_rad_s: float | None = None
    settle_hold_s: float | None = None
    max_final_attitude_error_deg: float | None = None
    max_rms_attitude_error_deg: float | None = None
    max_peak_attitude_error_deg: float | None = None
    max_final_body_rate_norm_rad_s: float | None = None
    max_body_rate_norm_rad_s: float | None = None
    max_settle_time_s: float | None = None
    max_total_torque_norm_n_m_s: float | None = None
    max_final_relative_distance_km: float | None = None
    max_final_range_km: float | None = None
    max_rms_relative_distance_km: float | None = None
    max_final_relative_speed_km_s: float | None = None
    max_final_speed_km_s: float | None = None
    max_radial_error_km: float | None = None
    max_intrack_error_km: float | None = None
    max_crosstrack_error_km: float | None = None
    max_radial_rate_km_s: float | None = None
    max_intrack_rate_km_s: float | None = None
    max_crosstrack_rate_km_s: float | None = None
    max_time_inside_keepout_s: float | None = None
    max_total_dv_m_s: float | None = None
    max_fuel_used_kg: float | None = None
    target_relative_state_ric_m: tuple[float, float, float, float, float, float] | None = None
    max_final_position_error_m: float | None = None
    max_final_velocity_error_m_s: float | None = None
    waypoints_ric_m: tuple[tuple[float, float, float, float, float, float], ...] = ()
    waypoint_tolerance_m: float | None = None
    waypoint_velocity_tolerance_m_s: float | None = None
    min_final_range_km: float | None = None
    min_range_km: float | None = None
    max_range_km: float | None = None
    target_delta_v_m_s: float | None = None
    max_delta_v_error_m_s: float | None = None
    target_semimajor_axis_km: float | None = None
    max_semimajor_axis_error_km: float | None = None
    target_eccentricity: float | None = None
    max_eccentricity_error: float | None = None
    max_navigation_position_error_m: float | None = None
    max_navigation_velocity_error_m_s: float | None = None
    max_actuator_saturation_fraction: float | None = None
    max_realization_force_error_n: float | None = None
    max_realization_torque_error_n_m: float | None = None
    min_accepted_command_fraction: float | None = None
    min_physical_realization_fraction: float | None = None
    min_audit_completeness_fraction: float | None = None
    max_command_count: int | None = None
    max_missed_task_releases: int | None = None
    max_missed_sensor_releases: int | None = None
    expected_translation_mode: str = ""
    expected_control_law: str = ""
    min_invocation_count: int | None = None
    max_cadence_jitter_s: float | None = None
    target_altitude_km: float | None = None
    max_final_apoapsis_error_km: float | None = None
    max_final_periapsis_error_km: float | None = None
    max_final_eccentricity: float | None = None
    max_insertion_time_s: float | None = None
    max_dynamic_pressure_pa: float | None = None
    max_mach: float | None = None
    max_abs_alpha_deg: float | None = None
    max_tvc_gimbal_deg: float | None = None
    max_aero_force_n: float | None = None
    min_propellant_remaining_fraction: float | None = None
    require_insertion: bool = False
    require_not_terminated_early: bool = False
    require_preemption: bool = False
    require_recovery: bool = False
    require_primary_resume: bool = False


@dataclass(frozen=True)
class ControllerBenchEngineeringConfig:
    """Engineering evidence emitted around each deterministic bench run."""

    enabled: bool = True
    semantic_lint: bool = True
    stack_conformance: bool = True
    frame_sign_probe: bool = True
    cadence_analysis: bool = True
    aligned_timeline: bool = True
    authority_analysis: bool = True
    stability_analysis: bool = True
    failure_reduction: bool = True
    evidence_manifest: bool = True
    timeline_max_rows: int = 20000


@dataclass(frozen=True)
class ControllerBenchCase:
    name: str
    config_path: Path
    description: str = ""
    metrics: tuple[ControllerBenchMetric, ...] = ()
    pass_criteria: tuple[ControllerBenchPassCriterion, ...] = ()
    objectives: tuple[ControllerBenchObjective, ...] = ()


@dataclass(frozen=True)
class ControllerBenchConfig:
    suite_name: str
    description: str = ""
    output_dir: Path = Path("outputs/controller_bench")
    plot_mode: str = "save"
    controller_target: ControllerBenchTarget = field(
        default_factory=lambda: ControllerBenchTarget(object_id="target", slot="flight_software")
    )
    variants: tuple[ControllerVariant, ...] = ()
    cases: tuple[ControllerBenchCase, ...] = ()
    metrics: tuple[ControllerBenchMetric, ...] = ()
    pass_criteria: tuple[ControllerBenchPassCriterion, ...] = ()
    objectives: tuple[ControllerBenchObjective, ...] = ()
    save_run_payloads: bool = True
    disable_plots: bool = True
    disable_animations: bool = True
    print_individual_run_summaries: bool = False
    parallel_enabled: bool = False
    parallel_workers: int = 0
    parallel_backend: str = "process_pool"
    optimization: ControllerBenchOptimizationConfig = field(default_factory=ControllerBenchOptimizationConfig)
    envelope: ControllerBenchEnvelopeConfig = field(default_factory=ControllerBenchEnvelopeConfig)
    envelope_manifest: dict[str, Any] = field(default_factory=dict)
    engineering: ControllerBenchEngineeringConfig = field(default_factory=ControllerBenchEngineeringConfig)
