from __future__ import annotations

import math

import numpy as np
import pytest

from sim.orbital_calculator import (
    circular_orbit_from_altitude,
    mission_recovery_from_intrack_impulse,
    repeat_ground_track_approximation,
    rocket_equation_mass_ratio,
    rv_to_robust_elements,
)


def _circular_state() -> tuple[float, float, float, float, float, float]:
    radius_km = 7000.0
    speed_km_s = math.sqrt(398600.4418 / radius_km)
    return (radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_public_orbital_calculators_reject_boolean_numeric_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        circular_orbit_from_altitude(value)
    with pytest.raises(ValueError):
        repeat_ground_track_approximation(400.0, value)
    with pytest.raises(ValueError):
        rocket_equation_mass_ratio(value, 300.0)
    with pytest.raises(ValueError):
        mission_recovery_from_intrack_impulse(
            400.0,
            -5.0,
            100.0,
            220.0,
            max_phasing_orbits=value,
        )


def test_optional_body_radius_and_standard_gravity_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="body_radius_km"):
        rv_to_robust_elements(*_circular_state(), body_radius_km=math.nan)
    with pytest.raises(ValueError, match="body_radius_km"):
        circular_orbit_from_altitude(400.0, body_radius_km=math.inf)
    with pytest.raises(ValueError, match="standard_gravity_m_s2"):
        rocket_equation_mass_ratio(100.0, 300.0, standard_gravity_m_s2=math.nan)
    with pytest.raises(ValueError, match="standard_gravity_m_s2"):
        rocket_equation_mass_ratio(100.0, 300.0, standard_gravity_m_s2=0.0)
