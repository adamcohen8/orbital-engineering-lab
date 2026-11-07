from __future__ import annotations

import math

import numpy as np
import pytest

from sim.dynamics.orbit.elements import coe_to_rv_eci
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2
from sim.dynamics.orbit.lambert import solve_lambert_universal_variable
from sim.dynamics.orbit.two_body import propagate_two_body_rk4


def _mean_anomaly_from_true(nu_rad: float, ecc: float) -> float:
    if ecc <= 1.0e-12:
        return float(nu_rad % (2.0 * math.pi))
    e_anom = math.atan2(math.sqrt(1.0 - ecc * ecc) * math.sin(nu_rad), ecc + math.cos(nu_rad))
    return float((e_anom - ecc * math.sin(e_anom)) % (2.0 * math.pi))


def _time_of_flight_between_true_anomalies(
    *,
    a_km: float,
    ecc: float,
    true_anomaly_1_deg: float,
    true_anomaly_2_deg: float,
) -> float:
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (float(a_km) ** 3))
    m1 = _mean_anomaly_from_true(math.radians(float(true_anomaly_1_deg)), float(ecc))
    m2 = _mean_anomaly_from_true(math.radians(float(true_anomaly_2_deg)), float(ecc))
    return float(((m2 - m1) % (2.0 * math.pi)) / mean_motion)


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=float).reshape(3)
    unit_axis = unit_axis / float(np.linalg.norm(unit_axis))
    vec = np.asarray(vector, dtype=float).reshape(3)
    return (
        vec * math.cos(float(angle_rad))
        + np.cross(unit_axis, vec) * math.sin(float(angle_rad))
        + unit_axis * float(np.dot(unit_axis, vec)) * (1.0 - math.cos(float(angle_rad)))
    )


def _optimal_hohmann_plane_change_split_rad(
    *,
    circular_departure_v_km_s: float,
    transfer_perigee_v_km_s: float,
    transfer_apogee_v_km_s: float,
    circular_arrival_v_km_s: float,
    plane_change_rad: float,
) -> float:
    lo = 0.0
    hi = float(plane_change_rad)

    def total_delta_v(split_rad: float) -> float:
        arrival_split_rad = float(plane_change_rad) - float(split_rad)
        return math.sqrt(
            circular_departure_v_km_s**2
            + transfer_perigee_v_km_s**2
            - 2.0 * circular_departure_v_km_s * transfer_perigee_v_km_s * math.cos(float(split_rad))
        ) + math.sqrt(
            circular_arrival_v_km_s**2
            + transfer_apogee_v_km_s**2
            - 2.0 * circular_arrival_v_km_s * transfer_apogee_v_km_s * math.cos(arrival_split_rad)
        )

    for _ in range(120):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if total_delta_v(left) < total_delta_v(right):
            hi = right
        else:
            lo = left
    return float(0.5 * (lo + hi))


def _assert_lambert_recovers_keplerian_arc(
    *,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_1_deg: float,
    true_anomaly_2_deg: float,
    velocity_tolerance_m_s: float = 1.0e-3,
) -> None:
    r1, v1 = coe_to_rv_eci(
        a_km=a_km,
        ecc=ecc,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_1_deg,
    )
    r2, v2 = coe_to_rv_eci(
        a_km=a_km,
        ecc=ecc,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_2_deg,
    )
    time_of_flight_s = _time_of_flight_between_true_anomalies(
        a_km=a_km,
        ecc=ecc,
        true_anomaly_1_deg=true_anomaly_1_deg,
        true_anomaly_2_deg=true_anomaly_2_deg,
    )

    solution = solve_lambert_universal_variable(r1, r2, time_of_flight_s, short_way=True)

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert np.linalg.norm(solution.v1_km_s - v1) * 1000.0 < velocity_tolerance_m_s
    assert np.linalg.norm(solution.v2_km_s - v2) * 1000.0 < velocity_tolerance_m_s


def _propagate_uncontrolled_state(state: np.ndarray, duration_s: float, *, max_step_s: float = 2.0) -> np.ndarray:
    out = np.asarray(state, dtype=float).reshape(6).copy()
    steps = max(int(math.ceil(abs(float(duration_s)) / float(max_step_s))), 1)
    step_s = float(duration_s) / float(steps)
    zero_accel = np.zeros(3, dtype=float)
    for _ in range(steps):
        out = propagate_two_body_rk4(out, step_s, EARTH_MU_KM3_S2, zero_accel)
    return out


def test_lambert_solves_circular_orbit_arc() -> None:
    a_km = 7000.0
    r1, v1 = coe_to_rv_eci(
        a_km=a_km,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=0.0,
        true_anomaly_deg=0.0,
    )
    r2, v2 = coe_to_rv_eci(
        a_km=a_km,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=0.0,
        true_anomaly_deg=60.0,
    )
    period_s = 2.0 * math.pi * math.sqrt(a_km**3 / EARTH_MU_KM3_S2)
    solution = solve_lambert_universal_variable(r1, r2, period_s / 6.0)

    assert solution.converged is True
    assert solution.iterations > 0
    assert abs(solution.residual_s) < 1.0e-5
    assert np.linalg.norm(solution.v1_km_s - v1) * 1000.0 < 1.0e-3
    assert np.linalg.norm(solution.v2_km_s - v2) * 1000.0 < 1.0e-3


@pytest.mark.parametrize(
    (
        "case_name",
        "a_km",
        "ecc",
        "inc_deg",
        "raan_deg",
        "argp_deg",
        "true_anomaly_1_deg",
        "true_anomaly_2_deg",
    ),
    [
        ("inclined eccentric short arc", 12000.0, 0.25, 50.0, 15.0, 25.0, 20.0, 110.0),
        ("eccentric wraparound short arc", 12000.0, 0.25, 50.0, 15.0, 25.0, 300.0, 40.0),
        ("high eccentric inclined short arc", 18000.0, 0.55, 63.4, 120.0, 45.0, 15.0, 135.0),
    ],
)
def test_lambert_recovers_known_keplerian_short_way_arc(
    case_name: str,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_1_deg: float,
    true_anomaly_2_deg: float,
) -> None:
    del case_name
    r1, v1 = coe_to_rv_eci(
        a_km=a_km,
        ecc=ecc,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_1_deg,
    )
    r2, v2 = coe_to_rv_eci(
        a_km=a_km,
        ecc=ecc,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_2_deg,
    )
    time_of_flight_s = _time_of_flight_between_true_anomalies(
        a_km=a_km,
        ecc=ecc,
        true_anomaly_1_deg=true_anomaly_1_deg,
        true_anomaly_2_deg=true_anomaly_2_deg,
    )

    solution = solve_lambert_universal_variable(r1, r2, time_of_flight_s, short_way=True)

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert np.linalg.norm(solution.v1_km_s - v1) * 1000.0 < 1.0e-3
    assert np.linalg.norm(solution.v2_km_s - v2) * 1000.0 < 1.0e-3


def test_lambert_recovers_known_circular_long_way_retrograde_arc() -> None:
    a_km = 7000.0
    r1, prograde_v1 = coe_to_rv_eci(
        a_km=a_km,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=25.0,
        true_anomaly_deg=0.0,
    )
    r2, prograde_v2 = coe_to_rv_eci(
        a_km=a_km,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=25.0,
        true_anomaly_deg=60.0,
    )
    period_s = 2.0 * math.pi * math.sqrt(a_km**3 / EARTH_MU_KM3_S2)

    solution = solve_lambert_universal_variable(r1, r2, 5.0 * period_s / 6.0, short_way=False)

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert np.linalg.norm(solution.v1_km_s + prograde_v1) * 1000.0 < 1.0e-3
    assert np.linalg.norm(solution.v2_km_s + prograde_v2) * 1000.0 < 1.0e-3


def test_lambert_short_and_long_way_branches_have_opposite_angular_momentum() -> None:
    radius_km = 7000.0
    transfer_angle_rad = math.radians(120.0)
    r1 = np.array([radius_km, 0.0, 0.0])
    r2 = np.array([radius_km * math.cos(transfer_angle_rad), radius_km * math.sin(transfer_angle_rad), 0.0])
    time_of_flight_s = 4000.0

    short_way = solve_lambert_universal_variable(r1, r2, time_of_flight_s, short_way=True)
    long_way = solve_lambert_universal_variable(r1, r2, time_of_flight_s, short_way=False)

    assert short_way.converged is True
    assert long_way.converged is True
    assert np.cross(r1, short_way.v1_km_s)[2] > 0.0
    assert np.cross(r1, long_way.v1_km_s)[2] < 0.0


@pytest.mark.parametrize(
    (
        "case_name",
        "a_km",
        "ecc",
        "inc_deg",
        "raan_deg",
        "argp_deg",
        "true_anomaly_1_deg",
        "true_anomaly_2_deg",
    ),
    [
        ("very high eccentricity outbound", 30000.0, 0.85, 40.0, 30.0, 70.0, 5.0, 120.0),
        ("extreme eccentricity outbound", 50000.0, 0.95, 40.0, 30.0, 70.0, 20.0, 150.0),
        ("high eccentricity inbound", 42000.0, 0.70, 40.0, 30.0, 70.0, 210.0, 330.0),
    ],
)
def test_lambert_recovers_high_eccentricity_keplerian_arcs(
    case_name: str,
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_1_deg: float,
    true_anomaly_2_deg: float,
) -> None:
    del case_name
    _assert_lambert_recovers_keplerian_arc(
        a_km=a_km,
        ecc=ecc,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_1_deg=true_anomaly_1_deg,
        true_anomaly_2_deg=true_anomaly_2_deg,
    )


@pytest.mark.parametrize(
    (
        "case_name",
        "a_km",
        "ecc",
        "true_anomaly_1_deg",
        "true_anomaly_2_deg",
    ),
    [
        ("large radius ratio outbound", 50000.0, 0.82, 0.0, 150.0),
        ("large radius ratio inbound", 50000.0, 0.82, 210.0, 330.0),
    ],
)
def test_lambert_recovers_large_radius_ratio_keplerian_arcs(
    case_name: str,
    a_km: float,
    ecc: float,
    true_anomaly_1_deg: float,
    true_anomaly_2_deg: float,
) -> None:
    del case_name
    _assert_lambert_recovers_keplerian_arc(
        a_km=a_km,
        ecc=ecc,
        inc_deg=47.0,
        raan_deg=83.0,
        argp_deg=31.0,
        true_anomaly_1_deg=true_anomaly_1_deg,
        true_anomaly_2_deg=true_anomaly_2_deg,
    )


def test_lambert_recovers_near_zero_and_near_pi_transfer_angles() -> None:
    _assert_lambert_recovers_keplerian_arc(
        a_km=7000.0,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=0.0,
        true_anomaly_1_deg=0.0,
        true_anomaly_2_deg=0.5,
    )
    _assert_lambert_recovers_keplerian_arc(
        a_km=7000.0,
        ecc=0.0,
        inc_deg=28.5,
        raan_deg=15.0,
        argp_deg=0.0,
        true_anomaly_1_deg=0.0,
        true_anomaly_2_deg=179.999,
    )


def test_lambert_rejects_exact_same_position_singular_geometry() -> None:
    r1_km = np.array([7000.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="singular"):
        solve_lambert_universal_variable(r1_km, r1_km, 100.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"r1_km": np.array([7000.0, np.nan, 0.0])}, "r1_km must be finite"),
        ({"r2_km": np.array([7000.0, 0.0])}, "r2_km must contain exactly three"),
        ({"max_iterations": 0}, "max_iterations must be a positive integer"),
        ({"tolerance_s": np.nan}, "tolerance_s must be finite and positive"),
    ],
)
def test_lambert_rejects_invalid_inputs_before_backend_dispatch(kwargs: dict, message: str) -> None:
    base = {
        "r1_km": np.array([7000.0, 0.0, 0.0]),
        "r2_km": np.array([0.0, 8000.0, 0.0]),
        "time_of_flight_s": 4200.0,
        "backend": "rust",
    }
    with pytest.raises(ValueError, match=message):
        solve_lambert_universal_variable(**{**base, **kwargs})


def test_lambert_handles_high_energy_short_time_of_flight() -> None:
    r1_km = np.array([7000.0, 0.0, 0.0])
    r2_km = np.array([0.0, 7000.0, 0.0])
    time_of_flight_s = 60.0

    solution = solve_lambert_universal_variable(r1_km, r2_km, time_of_flight_s)
    propagated = _propagate_uncontrolled_state(
        np.hstack((r1_km, solution.v1_km_s)),
        time_of_flight_s,
        max_step_s=0.5,
    )

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert np.isfinite(solution.v1_km_s).all()
    assert np.isfinite(solution.v2_km_s).all()
    assert np.linalg.norm(solution.v1_km_s) > 100.0
    assert np.linalg.norm(solution.v2_km_s) > 100.0
    assert np.linalg.norm(propagated[:3] - r2_km) < 1.0e-4
    assert np.linalg.norm(propagated[3:6] - solution.v2_km_s) * 1000.0 < 1.0e-3


def test_lambert_handles_long_zero_revolution_time_of_flight() -> None:
    r1_km = np.array([7000.0, 0.0, 0.0])
    r2_km = np.array([0.0, 7000.0, 0.0])

    solution = solve_lambert_universal_variable(r1_km, r2_km, 20000.0)

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert np.isfinite(solution.v1_km_s).all()
    assert np.isfinite(solution.v2_km_s).all()
    assert np.linalg.norm(solution.v1_km_s) < 20.0
    assert np.linalg.norm(solution.v2_km_s) < 20.0


def test_lambert_solution_reaches_endpoint_under_rk4_propagation() -> None:
    r1_km = np.array([7000.0, 0.0, 0.0])
    r2_km = np.array([0.0, 9000.0, 2500.0])
    time_of_flight_s = 1800.0
    solution = solve_lambert_universal_variable(r1_km, r2_km, time_of_flight_s)

    propagated = _propagate_uncontrolled_state(np.hstack((r1_km, solution.v1_km_s)), time_of_flight_s)

    assert solution.converged is True
    assert np.linalg.norm(propagated[:3] - r2_km) < 1.0e-3
    assert np.linalg.norm(propagated[3:6] - solution.v2_km_s) * 1000.0 < 2.0e-3


def test_lambert_recovers_seeded_random_keplerian_arc_set() -> None:
    rng = np.random.default_rng(20260708)
    for _ in range(12):
        a_km = float(rng.uniform(7200.0, 36000.0))
        ecc = float(rng.uniform(0.0, 0.65))
        true_anomaly_1_deg = float(rng.uniform(0.0, 360.0))
        true_anomaly_2_deg = true_anomaly_1_deg + float(rng.uniform(20.0, 160.0))
        _assert_lambert_recovers_keplerian_arc(
            a_km=a_km,
            ecc=ecc,
            inc_deg=float(rng.uniform(0.0, 120.0)),
            raan_deg=float(rng.uniform(0.0, 360.0)),
            argp_deg=float(rng.uniform(0.0, 360.0)),
            true_anomaly_1_deg=true_anomaly_1_deg,
            true_anomaly_2_deg=true_anomaly_2_deg,
            velocity_tolerance_m_s=2.0e-3,
        )


def test_lambert_approaches_combined_hohmann_arrival_plane_change() -> None:
    r1_km = 7000.0
    r2_km = 14000.0
    plane_change_deg = 30.0
    near_hohmann_angle_deg = 179.99
    transfer_a_km = 0.5 * (r1_km + r2_km)
    time_of_flight_s = math.pi * math.sqrt(transfer_a_km**3 / EARTH_MU_KM3_S2)

    theta_rad = math.radians(near_hohmann_angle_deg)
    r_depart_km = np.array([r1_km, 0.0, 0.0])
    r_arrive_km = np.array([r2_km * math.cos(theta_rad), r2_km * math.sin(theta_rad), 0.0])
    solution = solve_lambert_universal_variable(r_depart_km, r_arrive_km, time_of_flight_s)

    circular_departure_v_km_s = math.sqrt(EARTH_MU_KM3_S2 / r1_km)
    circular_arrival_v_km_s = math.sqrt(EARTH_MU_KM3_S2 / r2_km)
    transfer_perigee_v_km_s = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r1_km - 1.0 / transfer_a_km))
    transfer_apogee_v_km_s = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r2_km - 1.0 / transfer_a_km))
    expected_departure_burn_m_s = (transfer_perigee_v_km_s - circular_departure_v_km_s) * 1000.0
    expected_arrival_burn_m_s = (
        math.sqrt(
            circular_arrival_v_km_s**2
            + transfer_apogee_v_km_s**2
            - 2.0
            * circular_arrival_v_km_s
            * transfer_apogee_v_km_s
            * math.cos(math.radians(plane_change_deg))
        )
        * 1000.0
    )

    radial_hat = r_arrive_km / float(np.linalg.norm(r_arrive_km))
    transfer_tangent_hat = np.array([-math.sin(theta_rad), math.cos(theta_rad), 0.0])
    plane_normal_hat = np.cross(radial_hat, transfer_tangent_hat)
    target_tangent_hat = (
        transfer_tangent_hat * math.cos(math.radians(plane_change_deg))
        + plane_normal_hat * math.sin(math.radians(plane_change_deg))
    )
    initial_velocity_km_s = np.array([0.0, circular_departure_v_km_s, 0.0])
    target_velocity_km_s = circular_arrival_v_km_s * target_tangent_hat

    departure_burn_m_s = float(np.linalg.norm(solution.v1_km_s - initial_velocity_km_s) * 1000.0)
    arrival_burn_m_s = float(np.linalg.norm(target_velocity_km_s - solution.v2_km_s) * 1000.0)

    assert solution.converged is True
    assert abs(solution.residual_s) < 1.0e-5
    assert departure_burn_m_s == pytest.approx(expected_departure_burn_m_s, abs=1.0e-3)
    assert arrival_burn_m_s == pytest.approx(expected_arrival_burn_m_s, abs=1.0e-3)


def test_lambert_matches_optimal_split_plane_change_near_hohmann() -> None:
    r1_km = 7000.0
    r2_km = 14000.0
    plane_change_rad = math.radians(30.0)
    near_hohmann_angle_rad = math.radians(179.99)
    transfer_a_km = 0.5 * (r1_km + r2_km)
    time_of_flight_s = math.pi * math.sqrt(transfer_a_km**3 / EARTH_MU_KM3_S2)

    circular_departure_v_km_s = math.sqrt(EARTH_MU_KM3_S2 / r1_km)
    circular_arrival_v_km_s = math.sqrt(EARTH_MU_KM3_S2 / r2_km)
    transfer_perigee_v_km_s = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r1_km - 1.0 / transfer_a_km))
    transfer_apogee_v_km_s = math.sqrt(EARTH_MU_KM3_S2 * (2.0 / r2_km - 1.0 / transfer_a_km))
    departure_split_rad = _optimal_hohmann_plane_change_split_rad(
        circular_departure_v_km_s=circular_departure_v_km_s,
        transfer_perigee_v_km_s=transfer_perigee_v_km_s,
        transfer_apogee_v_km_s=transfer_apogee_v_km_s,
        circular_arrival_v_km_s=circular_arrival_v_km_s,
        plane_change_rad=plane_change_rad,
    )
    arrival_split_rad = plane_change_rad - departure_split_rad

    expected_departure_burn_m_s = (
        math.sqrt(
            circular_departure_v_km_s**2
            + transfer_perigee_v_km_s**2
            - 2.0
            * circular_departure_v_km_s
            * transfer_perigee_v_km_s
            * math.cos(departure_split_rad)
        )
        * 1000.0
    )
    expected_arrival_burn_m_s = (
        math.sqrt(
            circular_arrival_v_km_s**2
            + transfer_apogee_v_km_s**2
            - 2.0
            * circular_arrival_v_km_s
            * transfer_apogee_v_km_s
            * math.cos(arrival_split_rad)
        )
        * 1000.0
    )
    all_arrival_plane_change_m_s = (
        (transfer_perigee_v_km_s - circular_departure_v_km_s)
        + math.sqrt(
            circular_arrival_v_km_s**2
            + transfer_apogee_v_km_s**2
            - 2.0
            * circular_arrival_v_km_s
            * transfer_apogee_v_km_s
            * math.cos(plane_change_rad)
        )
    ) * 1000.0

    r_depart_km = np.array([r1_km, 0.0, 0.0])
    r_arrive_km = np.array(
        [
            r2_km * math.cos(near_hohmann_angle_rad),
            r2_km * math.sin(near_hohmann_angle_rad),
            0.0,
        ]
    )
    solution = solve_lambert_universal_variable(r_depart_km, r_arrive_km, time_of_flight_s)

    transfer_departure_hat = np.array([0.0, 1.0, 0.0])
    initial_velocity_hat = _rotate_about_axis(transfer_departure_hat, r_depart_km, -departure_split_rad)
    transfer_arrival_hat = np.array([-math.sin(near_hohmann_angle_rad), math.cos(near_hohmann_angle_rad), 0.0])
    target_velocity_hat = _rotate_about_axis(transfer_arrival_hat, r_arrive_km, arrival_split_rad)
    initial_velocity_km_s = circular_departure_v_km_s * initial_velocity_hat
    target_velocity_km_s = circular_arrival_v_km_s * target_velocity_hat

    departure_burn_m_s = float(np.linalg.norm(solution.v1_km_s - initial_velocity_km_s) * 1000.0)
    arrival_burn_m_s = float(np.linalg.norm(target_velocity_km_s - solution.v2_km_s) * 1000.0)
    total_burn_m_s = departure_burn_m_s + arrival_burn_m_s
    expected_total_burn_m_s = expected_departure_burn_m_s + expected_arrival_burn_m_s

    assert solution.converged is True
    assert math.degrees(departure_split_rad) == pytest.approx(5.100446, abs=1.0e-5)
    assert math.degrees(arrival_split_rad) == pytest.approx(24.899554, abs=1.0e-5)
    assert expected_total_burn_m_s < all_arrival_plane_change_m_s - 175.0
    assert departure_burn_m_s == pytest.approx(expected_departure_burn_m_s, abs=1.0e-3)
    assert arrival_burn_m_s == pytest.approx(expected_arrival_burn_m_s, abs=1.0e-3)
    assert total_burn_m_s == pytest.approx(expected_total_burn_m_s, abs=1.0e-3)


def test_lambert_rejects_exact_hohmann_plane_change_singular_geometry() -> None:
    r1_km = np.array([7000.0, 0.0, 0.0])
    r2_km = np.array([-14000.0, 0.0, 0.0])
    transfer_a_km = 10500.0
    time_of_flight_s = math.pi * math.sqrt(transfer_a_km**3 / EARTH_MU_KM3_S2)

    with pytest.raises(ValueError, match="singular"):
        solve_lambert_universal_variable(r1_km, r2_km, time_of_flight_s)


def test_lambert_rejects_invalid_time_of_flight() -> None:
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([0.0, 7000.0, 0.0])

    with pytest.raises(ValueError, match="time_of_flight_s"):
        solve_lambert_universal_variable(r1, r2, 0.0)
