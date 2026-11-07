"""Focused 2026-10-09 audit regressions for orbital frames and elements."""

import numpy as np
import pytest

from sim.dynamics.orbit.elements import (
    coe_to_rv_eci,
    coes_mapping_to_element_targets,
    rv_to_coe_eci,
)


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


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: coe_to_rv_eci(a_km=7000.0, ecc=np.nan, inc_deg=0.0, raan_deg=0.0,
                               argp_deg=0.0, true_anomaly_deg=0.0), "COE values must be finite"),
        (lambda: coe_to_rv_eci(a_km=7000.0, ecc=0.0, inc_deg=0.0, raan_deg=0.0,
                               argp_deg=0.0, true_anomaly_deg=0.0, mu_km3_s2=0.0),
         "COE mu_km3_s2 must be finite and positive"),
        (lambda: rv_to_coe_eci([7000.0, np.inf, 0.0], [0.0, 7.5, 0.0]),
         "Position and velocity must be finite"),
    ],
)
def test_orbital_element_converters_reject_nonfinite_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_element_targets_use_the_requested_gravitational_parameter() -> None:
    mu = 490280.0
    coes = {
        "a_km": 12000.0,
        "ecc": 0.2,
        "inc_deg": 15.0,
        "raan_deg": 20.0,
        "argp_deg": 30.0,
        "true_anomaly_deg": 40.0,
    }
    energy, eccentricity_vector, hhat = coes_mapping_to_element_targets(coes, mu_km3_s2=mu)
    position, velocity = coe_to_rv_eci(**coes, mu_km3_s2=mu)

    np.testing.assert_allclose(energy, 0.5 * np.dot(velocity, velocity) - mu / np.linalg.norm(position))
    np.testing.assert_allclose(
        eccentricity_vector,
        np.cross(velocity, np.cross(position, velocity)) / mu - position / np.linalg.norm(position),
    )
    np.testing.assert_allclose(hhat, np.cross(position, velocity) / np.linalg.norm(np.cross(position, velocity)))
