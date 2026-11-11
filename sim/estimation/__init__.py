"""Orbit determination and parameter-estimation helpers.

The package API is resolved lazily so ONP workers do not import every OD
backend (and their optional numerical dependencies) merely to construct a
lightweight estimator.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {}


def _export(module: str, *names: str) -> None:
    _EXPORTS.update((name, module) for name in names)


_export("sim.estimation.aoi", "AoITrackingEstimator")
_export("sim.estimation.attitude_ekf", "AttitudeEKFEstimator")
_export("sim.estimation.batch_least_squares", "BatchLeastSquaresResult", "solve_batch_least_squares")
_export("sim.estimation.covariance", "correlation_from_covariance", "covariance_from_jacobian")
_export(
    "sim.estimation.ground_station_od",
    "predict_ground_station_measurement",
    "normalize_ground_station_measurements",
    "solve_ground_station_measurement_od",
)
_export(
    "sim.estimation.initial_orbit",
    "IODHypothesis",
    "evaluate_position_triplet_iod",
    "generate_constrained_lambert_angle_hypotheses",
    "run_angles_only_iod",
    "solve_gauss_angles_only_iod",
    "solve_gibbs_iod",
    "solve_herrick_gibbs_iod",
    "solve_lambert_position_iod",
)
_export("sim.estimation.joint_ekf", "JointStateEKFEstimator")
_export("sim.estimation.joint_state", "JointStateEstimator")
_export(
    "sim.estimation.maneuver_detection",
    "EKFManeuverDetectionConfig",
    "EKFManeuverDetectionUpdate",
    "EKFManeuverDetector",
    "chi_square_threshold",
)
_export("sim.estimation.observations", "ObservationBatch")
_export(
    "sim.estimation.ogp_od",
    "OGP_OD_SCHEMA_VERSION",
    "OGP_PV_PACKET_CONTRACT",
    "fit_ogp_mean_elements_to_observations",
    "run_ogp_pv_packet",
)
_export(
    "sim.estimation.orbit_determination",
    "build_dynamics_od_quality_gates",
    "build_orbit_od_parameter_set",
    "selected_orbit_od_parameters",
    "solve_dynamics_orbit_determination",
)
_export("sim.estimation.orbit_ekf", "OrbitEKFEstimator")
_export("sim.estimation.orbit_ukf", "OrbitUKFEstimator")
_export("sim.estimation.parameters", "EstimatedParameter", "ParameterSet")
_export(
    "sim.estimation.productization",
    "OD_AGENT_EVIDENCE_PACKET_SCHEMA_VERSION",
    "OD_DATASET_MANIFEST_SCHEMA_VERSION",
    "OD_SOLUTION_SCHEMA_VERSION",
    "STANDARD_OD_REVIEW_QUERIES",
    "ODAdapterRegistration",
    "ODAdapterRegistry",
    "build_od_agent_evidence_packet",
    "build_od_dataset_manifest",
    "check_od_evidence_freshness",
    "compare_od_execution_parity",
    "default_od_adapter_registry",
    "normalize_od_solution",
    "safely_inspect_od_input",
    "write_standard_od_plots",
)
_export(
    "sim.estimation.relative_hcw_ekf",
    "HCWRelativeEKFEstimator",
    "HCWRelativeEKFUpdateDiagnostics",
    "SSJ2RelativeEKFEstimator",
    "hcw_measurement_dimension",
    "hcw_measurement_jacobian",
    "hcw_measurement_vector",
    "hcw_state_transition_matrix",
    "normalize_hcw_measurement_model",
)
_export(
    "sim.estimation.relative_od",
    "RelativeODArc",
    "RelativeODObservation",
    "build_relative_od_parameter_set",
    "compare_relative_od_models",
    "make_relative_od_arc",
    "normalize_relative_od_dynamics",
    "propagate_relative_od_state",
    "sample_relative_od_observations",
    "solve_batch_relative_od",
)
_export(
    "sim.estimation.relative_od_integrated",
    "INTEGRATED_RELATIVE_OD_SCHEMA_VERSION",
    "NonlinearRelativeTruth",
    "generate_nonlinear_relative_truth",
    "run_integrated_relative_od",
)
_export(
    "sim.estimation.relative_onp_variational",
    "ONPRelativeVariationalResult",
    "onp_relative_variational_propagate",
)
_export(
    "sim.estimation.relative_th_ekf",
    "THRelativeEKFEstimator",
    "THRelativeEKFUpdateDiagnostics",
    "YARelativeEKFEstimator",
    "th_propagate_relative_state",
    "th_relative_transition_matrix",
    "th_variational_propagate_relative_state_and_stm",
    "th_variational_transition_matrix",
    "ya_closed_form_propagate_relative_state_and_stm",
    "ya_closed_form_transition_matrix",
    "ya_normalized_state_transition_matrix",
)
_export("sim.estimation.residual_audit", "build_residual_audit")
_export("sim.estimation.residuals", "ric_position_residuals_m", "state_difference_residuals")
_export(
    "sim.estimation.sequential_od",
    "SEQUENTIAL_OD_SCHEMA_VERSION",
    "SequentialODConfig",
    "SequentialProcessNoise",
    "SequentialMeasurementModel",
    "predict_sequential_state_covariance",
    "run_sequential_orbit_od",
)
_export(
    "sim.estimation.sgp4_od",
    "DEFAULT_SGP4_SENSOR_FRAME_TRANSFORM",
    "fit_sgp4_mean_elements_to_ground_station_measurements",
    "fit_ogp_mean_elements_to_ground_station_measurements",
    "fit_sgp4_mean_elements_to_observations",
)
_export(
    "sim.estimation.slr_od",
    "fit_ogp_mean_elements_to_slr_normal_points",
    "fit_sgp4_mean_elements_to_slr_normal_points",
)
_export(
    "sim.estimation.weighting",
    "covariance_from_sigmas",
    "validate_covariance_block",
    "whiten_residual_block",
)

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
