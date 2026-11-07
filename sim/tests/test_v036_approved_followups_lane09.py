from __future__ import annotations

import numpy as np
import pytest

from sim.config import GroundStationSection
from sim.config.scenario.loader import load_simulation_yaml_bytes
from sim.estimation.ground_station_od import normalize_ground_station_measurements
from sim.ground_stations import (
    evaluate_ground_station_measurements,
    ground_station_measurements_to_observation_packet,
)
from sim.knowledge.object_tracking import KnowledgeConditionConfig
from sim.sensors.access import AccessConfig


def _measured_station(*, seed: int = 0, noise: dict[str, float] | None = None) -> GroundStationSection:
    return GroundStationSection(
        id="lane09-approved",
        lat_deg=0.0,
        lon_deg=0.0,
        measurements={
            "enabled": True,
            "measurement_type": "az_el_range",
            "update_cadence_s": 1.0,
            "seed": seed,
            "noise": dict(noise or {}),
        },
    )


def _truth() -> dict[str, np.ndarray]:
    return {
        "sat": np.asarray(
            [
                [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
                [7000.0, 7.5, 0.0, 0.0, 7.5, 0.0],
            ],
            dtype=float,
        )
    }


def test_invalid_gaussian_range_is_retained_and_rejected_before_conversion() -> None:
    measurements = evaluate_ground_station_measurements(
        ground_stations=[
            _measured_station(
                seed=2,
                noise={
                    "range_sigma_km": 2000.0,
                    "range_rate_sigma_km_s": 0.0,
                    "angle_sigma_deg": 0.0,
                },
            )
        ],
        t_s=np.asarray([0.0, 1.0]),
        truth_hist=_truth(),
    )
    target = measurements["lane09-approved"]["targets"]["sat"]
    assert target["measurement_count"] == 2
    assert target["usable_measurement_count"] == 0
    assert target["invalid_measurement_count"] == 2
    assert all(row["valid"] is False for row in target["measurements"])
    assert all(row["validity_reason"] == "range_nonpositive" for row in target["measurements"])
    assert all(row["range_km"] < 0.0 for row in target["measurements"])
    with pytest.raises(ValueError, match="physically invalid"):
        ground_station_measurements_to_observation_packet(
            ground_station_measurements=measurements,
            station_id="lane09-approved",
            object_id="sat",
        )


def test_zero_noise_draws_remain_unchanged() -> None:
    measurements = evaluate_ground_station_measurements(
        ground_stations=[_measured_station()],
        t_s=np.asarray([0.0, 1.0]),
        truth_hist=_truth(),
    )
    target = measurements["lane09-approved"]["targets"]["sat"]
    assert target["usable_measurement_count"] == target["measurement_count"] == 2
    assert all(row["valid"] is True for row in target["measurements"])
    assert all(row["validity_reason"] == "valid" for row in target["measurements"])
    assert target["measurements"][0]["range_km"] == pytest.approx(621.863, rel=1.0e-4)


def test_ground_station_od_rejects_impossible_domain_without_validity_flag() -> None:
    with pytest.raises(ValueError, match="physically invalid"):
        normalize_ground_station_measurements(
            [
                {
                    "time_jd_utc": 2460000.5,
                    "station_id": "lane09-approved",
                    "station_metadata": {"lat_deg": 0.0, "lon_deg": 0.0, "alt_km": 0.0},
                    "components": ["azimuth_deg", "elevation_deg", "range_km"],
                    "vector": [0.0, 91.0, 10.0],
                    "sigma": [0.1, 0.1, 0.1],
                }
            ]
        )


def test_yaml_loader_rejects_dual_fov_forms_before_runtime_normalization(tmp_path) -> None:
    content = b"""
schema_version: oel.scenario.v1
objects:
  target:
    enabled: true
    knowledge:
      conditions:
        fov_half_angle_rad: 0.2
        solid_angle_sr: 1.0
simulator:
  duration_s: 1.0
  dt_s: 1.0
"""
    with pytest.raises(
        ValueError,
        match=r"objects\.target\.knowledge\.conditions\.fov_half_angle_rad.*solid_angle_sr.*mutually exclusive",
    ):
        load_simulation_yaml_bytes(
            content,
            source_path=tmp_path / "dual-fov.yaml",
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AccessConfig(fov_half_angle_rad=0.2, solid_angle_sr=1.0),
        lambda: KnowledgeConditionConfig(fov_half_angle_rad=0.2, solid_angle_sr=1.0),
        lambda: KnowledgeConditionConfig.from_knowledge(
            {
                "conditions": {
                    "fov_half_angle_rad": 0.2,
                    "solid_angle_sr": 1.0,
                }
            },
            default_period_s=1.0,
        ),
    ],
)
def test_fov_forms_are_mutually_exclusive(factory) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        factory()


@pytest.mark.parametrize(
    "config",
    [
        AccessConfig(fov_half_angle_rad=0.2),
        AccessConfig(solid_angle_sr=1.0),
        KnowledgeConditionConfig(fov_half_angle_rad=0.2),
        KnowledgeConditionConfig(solid_angle_sr=1.0),
    ],
)
def test_each_fov_form_remains_valid_when_configured_alone(config) -> None:
    assert config is not None
