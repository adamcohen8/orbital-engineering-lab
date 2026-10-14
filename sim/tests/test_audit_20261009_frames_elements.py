"""Focused 2026-10-09 audit regressions for orbital frames and elements."""

import numpy as np
import pytest

from sim.dynamics.orbit.elements import coe_to_rv_eci, rv_to_coe_eci


@pytest.mark.parametrize("inclination_deg", [0.0, 180.0])
def test_equatorial_eccentric_coe_round_trip_preserves_periapsis_longitude(
    inclination_deg: float,
) -> None:
    """A zero-node orbit must retain its in-plane periapsis orientation."""

    original = {
        "a_km": 7000.0,
        "ecc": 0.2,
        "inc_deg": inclination_deg,
        "raan_deg": 11.0,
        "argp_deg": 25.0,
        "true_anomaly_deg": 33.0,
    }
    position, velocity = coe_to_rv_eci(**original)

    recovered = rv_to_coe_eci(position, velocity)
    expected_argp = (
        original["raan_deg"] + original["argp_deg"]
        if inclination_deg == 0.0
        else original["argp_deg"] - original["raan_deg"]
    ) % 360.0
    np.testing.assert_allclose(recovered.raan_deg, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(recovered.argp_deg, expected_argp, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        recovered.true_anomaly_deg,
        original["true_anomaly_deg"],
        rtol=0.0,
        atol=1.0e-12,
    )
    round_trip_position, round_trip_velocity = coe_to_rv_eci(
        a_km=recovered.a_km,
        ecc=recovered.ecc,
        inc_deg=recovered.inc_deg,
        raan_deg=recovered.raan_deg,
        argp_deg=recovered.argp_deg,
        true_anomaly_deg=recovered.true_anomaly_deg,
    )

    np.testing.assert_allclose(round_trip_position, position, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(round_trip_velocity, velocity, rtol=0.0, atol=1.0e-11)


def test_retrograde_equatorial_circular_coe_uses_reverse_true_longitude() -> None:
    position, velocity = coe_to_rv_eci(
        a_km=7000.0,
        ecc=0.0,
        inc_deg=180.0,
        raan_deg=11.0,
        argp_deg=25.0,
        true_anomaly_deg=33.0,
    )

    recovered = rv_to_coe_eci(position, velocity)

    np.testing.assert_allclose(recovered.raan_deg, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(recovered.argp_deg, 0.0, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(recovered.true_anomaly_deg, 47.0, rtol=0.0, atol=1.0e-12)

    round_trip_position, round_trip_velocity = coe_to_rv_eci(
        a_km=recovered.a_km,
        ecc=recovered.ecc,
        inc_deg=recovered.inc_deg,
        raan_deg=recovered.raan_deg,
        argp_deg=recovered.argp_deg,
        true_anomaly_deg=recovered.true_anomaly_deg,
    )
    np.testing.assert_allclose(round_trip_position, position, rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(round_trip_velocity, velocity, rtol=0.0, atol=1.0e-11)
