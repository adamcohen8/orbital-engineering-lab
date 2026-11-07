from __future__ import annotations

import numpy as np

from sim.dynamics.aerodynamic_effectors import (
    AerodynamicSurfaceGeometry,
    VariableGeometryAerodynamics,
    VariableGeometryAerodynamicsConfig,
)
from sim.dynamics.coupled_satellite import CoupledSatelliteState


def _state() -> CoupledSatelliteState:
    return CoupledSatelliteState(
        np.array([6_678.0, 0.0, 0.0]),
        np.array([0.0, 7.7, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        5_000.0,
        np.zeros(2),
        0.0,
    )


def test_realized_effector_geometry_changes_stage_force_torque_and_mass_properties() -> None:
    model = VariableGeometryAerodynamics(
        VariableGeometryAerodynamicsConfig(
            (
                AerodynamicSurfaceGeometry("flap", 20.0, 2.2, 0.5, (0.0, 1.0, 0.0), surface_mass_kg=5.0),
            ),
            bank_actuator_id="bank",
            base_drag_area_m2=5.0,
        )
    )
    retracted, retracted_review = model.evaluate(
        _state(), density_kg_m3=1.0e-10, device_positions={"flap": 0.0, "bank": 0.0}
    )
    deployed, deployed_review = model.evaluate(
        _state(), density_kg_m3=1.0e-10, device_positions={"flap": 1.0, "bank": 0.4}
    )
    assert np.linalg.norm(deployed.force_eci_n) > np.linalg.norm(retracted.force_eci_n)
    assert np.linalg.norm(deployed.torque_body_n_m) > 0.0
    assert deployed_review.drag_area_m2 > retracted_review.drag_area_m2
    assert deployed_review.bank_angle_rad == 0.4
    base_inertia = np.eye(3) * 100.0
    assert not np.array_equal(
        model.mass_properties(base_inertia, device_positions={"flap": 1.0}).inertia_body_kg_m2,
        base_inertia,
    )


def test_surface_torque_weights_each_surface_force_at_its_own_center() -> None:
    model = VariableGeometryAerodynamics(
        VariableGeometryAerodynamicsConfig(
            (
                AerodynamicSurfaceGeometry("left", 1.0, 1.0, 0.0, (1.0, 0.0, 0.0)),
                AerodynamicSurfaceGeometry("right", 1.0, 3.0, 0.0, (-1.0, 0.0, 0.0)),
            )
        )
    )

    effects, _ = model.evaluate(
        _state(),
        density_kg_m3=1.0e-10,
        device_positions={"left": 1.0, "right": 1.0},
    )

    # The two equal-area surfaces have unequal force weights.  Their resultant
    # drag is applied at a nonzero net moment arm, producing +z torque.
    dynamic_pressure = 0.5 * 1.0e-10 * (7_700.0**2)
    np.testing.assert_allclose(
        effects.torque_body_n_m,
        np.array([0.0, 0.0, 2.0 * dynamic_pressure]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_lift_reference_uses_a_nonparallel_fallback_for_z_axis_wind() -> None:
    model = VariableGeometryAerodynamics(
        VariableGeometryAerodynamicsConfig(
            (AerodynamicSurfaceGeometry("flap", 2.0, 2.0, 0.5, (1.0, 0.0, 0.0)),),
        )
    )
    state = CoupledSatelliteState(
        np.array([0.0, 0.0, 6_678.0]),
        np.array([0.0, 0.0, 7.7]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        5_000.0,
        np.zeros(2),
        0.0,
    )

    effects, _ = model.evaluate(state, density_kg_m3=1.0e-10, device_positions={"flap": 1.0})

    assert np.all(np.isfinite(effects.force_eci_n))
    assert np.all(np.isfinite(effects.torque_body_n_m))
    assert abs(float(effects.force_eci_n[0])) > 0.0
