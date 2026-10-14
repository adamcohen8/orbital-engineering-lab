"""Public coverage and ground-geometry audit regressions."""

from __future__ import annotations

import numpy as np
import pytest

from sim.analysis.directed_link import free_space_link_ledger
from sim.analysis.event_refinement import refine_availability_transitions
from sim.analysis.global_coverage import summarize_sampled_coverage_mask
from sim.analysis.observer_target_geometry import evaluate_surface_targets_ecef
from sim.analysis.sensor_footprint_geometry import intersect_rays_wgs84
from sim.ground_stations import _line_of_sight_from_ground
from sim.utils.geodesy import WGS84_A_KM, geodetic_to_ecef_km


def test_ground_los_uses_wgs84_quadratic_minimizer() -> None:
    """An oblate-Earth secant must be blocked even when Euclidean TCA misses it."""

    station = geodetic_to_ecef_km(-45.0, 0.0, 0.0)
    target = np.array([-8930.11033297, 240.25777577, -17852.51557766])

    assert _line_of_sight_from_ground(station, target) is False


def test_ground_los_rejects_target_inside_wgs84() -> None:
    station = geodetic_to_ecef_km(0.0, 0.0, 0.0)

    assert _line_of_sight_from_ground(station, np.array([6300.0, 0.0, 0.0])) is False


@pytest.mark.parametrize("latitude_deg", [-90.0, -60.0, -45.0, 0.0, 45.0, 60.0, 90.0])
def test_ground_los_accepts_outward_targets_at_surface_roundoff_boundaries(latitude_deg: float) -> None:
    station = geodetic_to_ecef_km(latitude_deg, 0.0, 0.0)
    target = np.asarray(station, dtype=float) * 1.1

    assert _line_of_sight_from_ground(station, target) is True


def test_rust_coverage_validates_integer_cell_indices_before_dispatch(monkeypatch) -> None:
    import sim.rust_coverage_backend as rust_backend

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("malformed input reached the native coverage backend")

    monkeypatch.setattr(rust_backend, "summarize_sampled_mask", unexpected_dispatch)
    with pytest.raises(ValueError, match="cell_indices must be an integer vector"):
        summarize_sampled_coverage_mask(
            np.array([[True], [False]], dtype=bool),
            np.array([0.0, 1.0]),
            cell_indices=np.array([0.5]),
            numeric_backend="rust",
        )


def test_rust_surface_geometry_validates_before_dispatch(monkeypatch) -> None:
    import sim.rust_coverage_backend as rust_backend

    monkeypatch.setattr(
        rust_backend,
        "surface_targets_ecef",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("malformed input reached the native surface backend")
        ),
    )
    with pytest.raises(ValueError, match="boresight_ecef must be a unit vector"):
        evaluate_surface_targets_ecef(
            observer_ecef_km=np.array([7000.0, 0.0, 0.0]),
            target_ecef_km=np.array([[WGS84_A_KM, 0.0, 0.0]]),
            target_outward_normal_ecef=np.array([[1.0, 0.0, 0.0]]),
            boresight_ecef=np.array([0.0, 0.0, 2.0]),
            half_angle_rad=0.2,
            numeric_backend="rust",
        )


def test_rust_ray_intersection_validates_before_dispatch(monkeypatch) -> None:
    import sim.rust_coverage_backend as rust_backend

    monkeypatch.setattr(
        rust_backend,
        "intersect_rays_wgs84",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("malformed input reached the native ray backend")),
    )
    with pytest.raises(ValueError, match="outside the WGS84 ellipsoid"):
        intersect_rays_wgs84(
            np.array([WGS84_A_KM, 0.0, 0.0]),
            np.array([[1.0, 0.0, 0.0]]),
            numeric_backend="rust",
        )


def test_rust_link_ledger_validates_before_dispatch(monkeypatch) -> None:
    import sim.rust_coverage_backend as rust_backend

    monkeypatch.setattr(
        rust_backend,
        "free_space_link_ledger",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("malformed input reached the native RF backend")),
    )
    with pytest.raises(ValueError, match="range_km must contain positive finite"):
        free_space_link_ledger(
            np.array([0.0]),
            carrier_frequency_hz=2.2e9,
            tx_power_w=10.0,
            tx_gain_dbi=6.0,
            rx_gain_dbi=12.0,
            data_rate_bps=1.0e6,
            system_noise_temperature_k=500.0,
            required_eb_n0_db=9.6,
            numeric_backend="rust",
        )


def test_rust_transition_refinement_validates_before_dispatch(monkeypatch) -> None:
    import sim.rust_coverage_backend as rust_backend

    monkeypatch.setattr(
        rust_backend,
        "refine_availability_transitions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("malformed input reached the native refinement backend")
        ),
    )
    with pytest.raises(ValueError, match="times_s must be strictly increasing"):
        refine_availability_transitions(
            np.array([0.0, 0.0]),
            np.array([False, True]),
            ("blocked", "available"),
            numeric_backend="rust",
        )
