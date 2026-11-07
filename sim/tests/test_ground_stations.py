from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import SimulationConfig, SimulationSession
from sim.config import GroundStationSection, scenario_config_from_dict
from sim.dynamics.orbit.epoch import datetime_to_julian_date
from sim.ground_stations import (
    evaluate_ground_station_access,
    evaluate_ground_station_measurements,
)
from sim.reporting.ground_station_access_reports import (
    DEFAULT_ACCESS_REPORT_EPOCH_UTC,
    build_ground_station_access_report_views,
    extract_access_windows,
    render_ground_station_access_report,
    render_satellite_access_report,
)
from sim.utils.geodesy import enu_to_ecef_rotation, geodetic_to_ecef_km


def _ground_station_config(output_dir: Path) -> dict:
    return {
        "scenario_name": "ground_station_access_smoke",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "ground_stations": [
            {
                "id": "equator_prime",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
                "alt_km": 0.0,
                "min_elevation_deg": 10.0,
                "max_range_km": 1000.0,
            }
        ],
        "simulator": {
            "duration_s": 1.0,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": True, "save_full_log": True},
            "plots": {"enabled": False, "figure_ids": []},
            "animations": {"enabled": False, "types": []},
        },
    }


def test_ground_station_yaml_parses_list_and_mapping_forms(tmp_path: Path) -> None:
    cfg = scenario_config_from_dict(_ground_station_config(tmp_path))
    assert len(cfg.ground_stations) == 1
    assert cfg.ground_stations[0].id == "equator_prime"
    assert cfg.ground_stations[0].max_range_km == 1000.0
    assert cfg.ground_stations[0].measurements == {}

    mapped = _ground_station_config(tmp_path)
    mapped["ground_stations"] = {
        "hawaii": {
            "lat_deg": 19.8,
            "lon_deg": -155.5,
            "altitude_km": 4.2,
            "min_elevation_deg": 5.0,
        }
    }
    cfg2 = scenario_config_from_dict(mapped)
    assert cfg2.ground_stations[0].id == "hawaii"
    assert cfg2.ground_stations[0].alt_km == 4.2

    measured = _ground_station_config(tmp_path)
    measured["ground_stations"][0]["measurements"] = {
        "enabled": True,
        "measurement_type": "az_el_range_rate",
        "update_cadence_s": 60.0,
        "noise": {"range_sigma_km": 0.01, "range_rate_sigma_km_s": 1.0e-5, "angle_sigma_deg": 0.02},
    }
    cfg3 = scenario_config_from_dict(measured)
    assert cfg3.ground_stations[0].measurements["enabled"] is True
    assert cfg3.ground_stations[0].measurements["noise"]["range_sigma_km"] == 0.01


def test_ground_station_yaml_rejects_invalid_values(tmp_path: Path) -> None:
    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["lat_deg"] = 120.0
    with pytest.raises(ValueError, match="lat_deg must be between"):
        scenario_config_from_dict(cfg)

    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["max_range_km"] = -1.0
    with pytest.raises(ValueError, match="max_range_km must be positive"):
        scenario_config_from_dict(cfg)

    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["measurements"] = {"enabled": True, "range_sigma_km": -0.1}
    with pytest.raises(ValueError, match="range_sigma_km must be non-negative"):
        scenario_config_from_dict(cfg)


def test_ground_station_yaml_rejects_unsupported_fields(tmp_path: Path) -> None:
    cfg = _ground_station_config(tmp_path)
    cfg["ground_stations"][0]["velocity_ned_m_s"] = [0.0, 12.0, 0.0]
    cfg["ground_stations"][0]["frequency_hz"] = 2.2e9

    with pytest.raises(ValueError, match="unsupported field"):
        scenario_config_from_dict(cfg)


def test_ground_station_history_rejects_nonmonotonic_times_and_bad_shapes() -> None:
    station = GroundStationSection(id="equator_prime", lat_deg=0.0, lon_deg=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_ground_station_access(
            ground_stations=[station],
            t_s=np.array([0.0, 1.0, 0.5]),
            truth_hist={"sat": np.zeros((3, 6))},
        )
    with pytest.raises(ValueError, match="shape"):
        evaluate_ground_station_access(
            ground_stations=[station],
            t_s=np.array([0.0]),
            truth_hist={"sat": np.zeros(6)},
        )
    measured_station = GroundStationSection(
        id="measured",
        lat_deg=0.0,
        lon_deg=0.0,
        measurements={"enabled": True},
    )
    with pytest.raises(ValueError, match="shape"):
        evaluate_ground_station_measurements(
            ground_stations=[measured_station],
            t_s=np.array([0.0]),
            truth_hist={"sat": np.zeros((1, 3))},
        )


def test_ground_station_access_geometry_applies_los_elevation_and_range() -> None:
    station = GroundStationSection(
        id="equator_prime",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=10.0,
        max_range_km=1000.0,
    )
    t_s = np.array([0.0])

    visible_hist, visible_summary = evaluate_ground_station_access(
        ground_stations=[station],
        t_s=t_s,
        truth_hist={"sat": np.array([[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert visible_hist["equator_prime"]["targets"]["sat"]["access"] == [True]
    assert visible_hist["equator_prime"]["targets"]["sat"]["reason"] == ["ok"]
    assert visible_summary["equator_prime"]["sat"]["access_samples"] == 1

    blocked_hist, _ = evaluate_ground_station_access(
        ground_stations=[station],
        t_s=t_s,
        truth_hist={"sat": np.array([[-7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert blocked_hist["equator_prime"]["targets"]["sat"]["access"] == [False]
    assert blocked_hist["equator_prime"]["targets"]["sat"]["reason"] == ["line_of_sight"]

    short_range_station = GroundStationSection(
        id="equator_prime",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=0.0,
        max_range_km=100.0,
    )
    range_hist, _ = evaluate_ground_station_access(
        ground_stations=[short_range_station],
        t_s=t_s,
        truth_hist={"sat": np.array([[7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]])},
    )
    assert range_hist["equator_prime"]["targets"]["sat"]["access"] == [False]
    assert range_hist["equator_prime"]["targets"]["sat"]["reason"] == ["range"]


def test_midlatitude_ground_station_los_uses_wgs84_ellipsoid() -> None:
    station = GroundStationSection(
        id="midlatitude",
        lat_deg=45.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=0.0,
    )
    station_ecef = geodetic_to_ecef_km(45.0, 0.0, 0.0)
    enu_to_ecef = enu_to_ecef_rotation(45.0, 0.0)
    elevation_rad = np.deg2rad(0.1)
    target_ecef = station_ecef + enu_to_ecef @ (
        5000.0 * np.array([0.0, np.cos(elevation_rad), np.sin(elevation_rad)])
    )

    histories, _ = evaluate_ground_station_access(
        ground_stations=[station],
        t_s=np.array([0.0]),
        truth_hist={"sat": np.array([np.r_[target_ecef, 0.0, 0.0, 0.0]])},
    )

    target = histories["midlatitude"]["targets"]["sat"]
    assert target["line_of_sight"] == [True]
    assert target["access"] == [True]


def test_ground_station_measurements_emit_access_limited_sensor_rows() -> None:
    station = GroundStationSection(
        id="equator_prime",
        lat_deg=0.0,
        lon_deg=0.0,
        alt_km=0.0,
        min_elevation_deg=10.0,
        max_range_km=5000.0,
        measurements={
            "enabled": True,
            "measurement_type": "az_el_range_rate",
            "update_cadence_s": 60.0,
            "range_sigma_km": 0.0,
            "range_rate_sigma_km_s": 0.0,
            "angle_sigma_deg": 0.0,
        },
    )
    t_s = np.array([0.0, 30.0, 60.0])
    truth = {
        "sat": np.array(
            [
                [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
                [7000.0, 225.0, 0.0, 0.0, 7.5, 0.0],
                [7000.0, 450.0, 0.0, 0.0, 7.5, 0.0],
            ]
        )
    }

    measurements = evaluate_ground_station_measurements(
        ground_stations=[station],
        t_s=t_s,
        truth_hist=truth,
    )

    target = measurements["equator_prime"]["targets"]["sat"]
    assert target["measurement_count"] == 2
    assert target["skipped"]["cadence"] == 1
    first = target["measurements"][0]
    assert first["components"] == ["azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s"]
    assert first["range_km"] == pytest.approx(621.863, rel=1.0e-4)
    assert first["elevation_deg"] == pytest.approx(90.0)
    assert first["range_rate_km_s"] == pytest.approx(0.0, abs=1.0e-8)


def test_single_run_records_ground_station_access_payload(tmp_path: Path) -> None:
    result = SimulationSession.from_config(SimulationConfig.from_dict(_ground_station_config(tmp_path))).run()

    access = result.ground_station_access
    station = access["equator_prime"]
    target = station["targets"]["target"]
    assert target["access"][0] is True
    assert target["reason"][0] == "ok"
    assert target["range_km"][0] == pytest.approx(621.863, rel=1e-4)

    summary = result.summary["ground_station_access_summary"]["equator_prime"]["target"]
    assert summary["access_samples"] >= 1
    assert summary["first_access_time_s"] == 0.0
    assert summary["min_range_km"] == pytest.approx(621.863, rel=1e-4)

    assert result.summary["ground_station_access_report_epoch_utc"] == "2026-01-01T00:00:00Z"
    report_outputs = result.summary["ground_station_access_report_outputs"]
    by_satellite = Path(report_outputs["by_satellite"])
    by_station = Path(report_outputs["by_ground_station"])
    assert by_satellite.exists()
    assert by_station.exists()
    assert "2026-01-01T00:00:00Z" in by_satellite.read_text(encoding="utf-8")
    assert "equator_prime -> target" in by_station.read_text(encoding="utf-8")


def test_single_run_records_ground_station_measurement_payload(tmp_path: Path) -> None:
    raw = _ground_station_config(tmp_path)
    raw["simulator"]["duration_s"] = 120.0
    raw["simulator"]["dt_s"] = 60.0
    raw["ground_stations"][0]["max_range_km"] = 5000.0
    raw["ground_stations"][0]["measurements"] = {
        "enabled": True,
        "measurement_type": "az_el_range_rate",
        "update_cadence_s": 60.0,
    }

    result = SimulationSession.from_config(SimulationConfig.from_dict(raw)).run()

    measurements = result.ground_station_measurements
    target = measurements["equator_prime"]["targets"]["target"]
    assert target["measurement_count"] >= 1
    summary = result.summary["ground_station_measurement_summary"]["equator_prime"]
    assert summary["measurement_count"] == target["measurement_count"]


def test_ground_station_access_reports_use_configured_utc_epoch() -> None:
    jd_start = datetime_to_julian_date(DEFAULT_ACCESS_REPORT_EPOCH_UTC) + 1.0
    t_s = np.array([0.0, 10.0, 20.0, 30.0])
    target_payload = {
        "access": [False, True, True, False],
        "range_km": [None, 900.0, 800.0, None],
        "elevation_deg": [None, 20.0, 40.0, None],
    }

    windows = extract_access_windows(t_s=t_s, target_payload=target_payload, jd_utc_start=jd_start)

    assert windows == [
        {
            "start_index": 1,
            "end_index": 2,
            "start_time_s": 10.0,
            "end_time_s": 30.0,
            "duration_s": 20.0,
            "aos_utc": "2026-01-02T00:00:10Z",
            "los_utc": "2026-01-02T00:00:30Z",
            "min_range_km": 800.0,
            "max_elevation_deg": 40.0,
        }
    ]


def test_ground_station_access_report_views_support_both_orientations() -> None:
    views = build_ground_station_access_report_views(
        ground_station_access={
            "site_a": {
                "station": {"id": "site_a"},
                "targets": {
                    "sat_1": {
                        "access": [True, False],
                        "range_km": [700.0, 710.0],
                        "elevation_deg": [50.0, 45.0],
                    }
                },
            }
        },
        ground_station_access_summary={
            "site_a": {
                "sat_1": {
                    "access_duration_s": 10.0,
                    "first_access_time_s": 0.0,
                    "last_access_time_s": 0.0,
                    "min_range_km": 700.0,
                    "max_elevation_deg": 50.0,
                }
            }
        },
        t_s=np.array([0.0, 10.0]),
        initial_jd_utc=None,
    )

    assert "sat_1" in views["by_satellite"]
    assert "site_a" in views["by_satellite"]["sat_1"]["stations"]
    assert "site_a" in views["by_ground_station"]
    assert "sat_1" in views["by_ground_station"]["site_a"]["satellites"]
    station_summary = views["by_ground_station"]["site_a"]["satellites"]["sat_1"]["summary"]
    assert station_summary["first_access_utc"] == "2026-01-01T00:00:00Z"


def test_ground_station_access_report_summary_uses_window_los_time() -> None:
    views = build_ground_station_access_report_views(
        ground_station_access={
            "site_a": {
                "station": {"id": "site_a"},
                "targets": {
                    "sat_1": {
                        "access": [True, True, False],
                        "range_km": [700.0, 800.0, 900.0],
                        "elevation_deg": [30.0, 20.0, 5.0],
                    }
                },
            }
        },
        ground_station_access_summary={
            "site_a": {
                "sat_1": {
                    "access_duration_s": 240.0,
                    "first_access_time_s": 0.0,
                    "last_access_time_s": 120.0,
                    "min_range_km": 700.0,
                    "max_elevation_deg": 30.0,
                }
            }
        },
        t_s=np.array([0.0, 120.0, 240.0]),
        initial_jd_utc=None,
    )

    by_station = render_ground_station_access_report(views)
    by_satellite = render_satellite_access_report(views)

    assert "Last LOS UTC" in by_station
    assert "Last Access UTC" not in by_station
    assert "| `sat_1` | 1 | 240.0 | 2026-01-01T00:00:00Z | 2026-01-01T00:04:00Z |" in by_station
    assert "| 2026-01-01T00:00:00Z | 2026-01-01T00:04:00Z | 240.0 |" in by_station
    assert "| `site_a` | 1 | 240.0 | 2026-01-01T00:00:00Z | 2026-01-01T00:04:00Z |" in by_satellite
