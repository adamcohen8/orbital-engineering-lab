"""Synthetic data generation utilities for OEL workflows.

The data-generation package is intentionally separate from scale processing so
synthetic truth catalogs, observation streams, and scoring fixtures can support
catalog screening, OD, AI training datasets, validation, reports, and future
workflows.
"""

from sim.data_generation.ground_station_synthetic import (
    angle_delta_deg,
    ground_station_measurement_count,
    ground_station_measurement_site,
    offset_cartesian_state,
    state_comparison_payload,
    tle_element_comparison_payload,
    tle_line2_with_mean_fields,
    truth_arc_tracking_network_from_eci_history,
    truth_arc_tracking_network_from_ogp,
    truth_arc_tracking_network_from_sgp4,
)
from sim.data_generation.workflow import generate_synthetic_data, plan_synthetic_data, score_synthetic_data

__all__ = [
    "angle_delta_deg",
    "ground_station_measurement_count",
    "ground_station_measurement_site",
    "generate_synthetic_data",
    "offset_cartesian_state",
    "plan_synthetic_data",
    "score_synthetic_data",
    "state_comparison_payload",
    "tle_element_comparison_payload",
    "tle_line2_with_mean_fields",
    "truth_arc_tracking_network_from_eci_history",
    "truth_arc_tracking_network_from_sgp4",
    "truth_arc_tracking_network_from_ogp",
]
