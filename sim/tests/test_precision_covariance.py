"""Complete parameter correlations, time coupling, process and ingestion guards."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sim.pro_tracking_od import precision_covariance as owner


@pytest.fixture
def design(monkeypatch):
    rng = np.random.default_rng(23)
    g = rng.normal(size=(3, 6, 13))
    scales = np.r_[[1e-3]*3, [1e-6]*3, [1e-12]*7]
    # Strong state/radiation cross-correlation; diagonalizing must change output.
    a = rng.normal(size=(13, 13))
    p = (a @ a.T) * scales[:, None] * scales[None, :]
    values = np.zeros(13)
    scaled_g = g / scales[None, None, :]
    class Predictor:
        def __init__(self, model, times):
            if len(times) != 3:
                raise ValueError("test design epoch count")
        def predict(self, state, coefficient, *, ecom_coefficients_km_s2):
            return np.einsum("nij,j->ni", scaled_g, np.r_[state, ecom_coefficients_km_s2] - values)
    monkeypatch.setattr(owner, "PrecisionOrbitPredictor", Predictor)
    model = SimpleNamespace(identity=lambda: {"test": True}, radiation_model="ecom2")
    solution = dict(schema=owner.SCHEMA, model=model.identity(), solver=dict(success=True),
                    covariance_evidence=dict(status="formal_local_gaussian", scale_policy="absolute"),
                    covariance=p.tolist(), initial_state_eme2000_km_km_s=values[:6].tolist(),
                    ecom2_coefficients_km_s2=values[6:].tolist(), rejected_observations=[],
                    covariance_coordinates="delta_state_and_ecom2", training_end_s=0.)
    return solution, model, scaled_g, p


def test_complete_cross_terms_and_joint_projection(design):
    solution, model, g, p = design
    f = owner.forecast_precision_covariance(solution, model=model, times_s=[1., 2., 3.])
    np.testing.assert_allclose([r["sensitivity_native"] for r in f["records"]], g, rtol=2e-8, atol=1e-8)
    expected = np.array([v @ p @ v.T for v in g])
    np.testing.assert_allclose([r["covariance_eci_km_km_s"] for r in f["records"]], expected, rtol=3e-8)
    assert not np.allclose(expected, [v @ np.diag(np.diag(p)) @ v.T for v in g])
    h = np.tile(np.eye(6)[:2], (3, 1, 1))
    # Includes observation cross-time covariance, independently of orbit errors.
    r = np.eye(6) * .2 + np.ones((6, 6)) * .01
    projection = owner.project_precision_covariance(f, measurement_jacobians=h, observation_covariance=r)
    jac = np.einsum("nij,njk->nik", h, g).reshape(6, 13)
    np.testing.assert_allclose(projection["covariance"], jac @ p @ jac.T + r, rtol=3e-8)
    assert abs(projection["parameter_component"][0][4]) > 1e-3
    assert not f["empirically_calibrated"] and not f["unknown_force_uncertainty_included"]


def test_joint_process_retains_shared_increments_and_matches_analytic_transport():
    times = np.array([0., 2., 5.])
    phi = np.tile(np.eye(6), (3, 1, 1))
    phi[:, :3, 3:] = times[:, None, None] * np.eye(3)
    def local(dt):
        return np.block([[np.eye(3)*dt**3/3, np.eye(3)*dt**2/2],
                         [np.eye(3)*dt**2/2, np.eye(3)*dt]])
    q = owner.precision_joint_process_covariance(times, phi, start_time_s=0., local_covariance=local)
    np.testing.assert_allclose(q[12:, 12:], local(5.), atol=1e-14)
    transport = np.eye(6)
    transport[:3, 3:] = 3*np.eye(3)
    np.testing.assert_allclose(q[6:12, 12:], local(2.) @ transport.T)
    assert np.linalg.eigvalsh(q).min() > -1e-12
    with pytest.raises(ValueError, match="last-assimilated"):
        owner.precision_joint_process_covariance(times, phi, start_time_s=1., local_covariance=local)


def test_process_projection_and_identity_guards(design):
    solution, model, g, p = design
    q = np.kron(np.minimum.outer([1., 2., 3.], [1., 2., 3.]), np.eye(6)) * .01
    f = owner.forecast_precision_covariance(solution, model=model, times_s=[1., 2., 3.],
                                           process_noise_joint_covariance=q)
    h = np.tile(np.eye(6), (3, 1, 1))
    result = owner.project_precision_covariance(f, measurement_jacobians=h, observation_covariance=np.eye(18))
    np.testing.assert_allclose(result["process_component"], q)
    assert f["unknown_force_uncertainty_included"]
    f["records"][0]["time_s"] = 9.
    with pytest.raises(ValueError, match="drift"):
        owner.project_precision_covariance(f, measurement_jacobians=h, observation_covariance=np.eye(18))


def test_multiple_stations_at_same_epoch_share_orbit_and_process_error(design):
    solution, model, g, p = design
    q=np.kron(np.minimum.outer([1.,2.,3.],[1.,2.,3.]),np.eye(6))*.01
    forecast=owner.forecast_precision_covariance(solution,model=model,times_s=[1.,2.,3.],process_noise_joint_covariance=q)
    h=np.array([[[1.,0.,0.,0.,0.,0.]],[[0.,1.,0.,0.,0.,0.]],[[1.,0.,0.,0.,0.,0.]]])
    indices=np.array([0,0,2])
    result=owner.project_precision_covariance(forecast,measurement_jacobians=h,epoch_indices=indices,observation_covariance=np.eye(3))
    selected=np.array([g[0,0],g[0,1],g[2,0]])
    coordinates=[0,1,12]
    np.testing.assert_allclose(result["covariance"],selected@p@selected.T+q[np.ix_(coordinates,coordinates)]+np.eye(3))
    with pytest.raises(ValueError,match="epoch index"):
        owner.project_precision_covariance(forecast,measurement_jacobians=h,epoch_indices=[0,.5,2],observation_covariance=np.eye(3))


@pytest.mark.parametrize("change", [dict(covariance=None), dict(solver=dict(success=False)),
    dict(rejected_observations=[0]), dict(covariance_coordinates="diagonal_only"),
    dict(covariance_evidence=dict(status="formal_local_gaussian", scale_policy="residual"))])
def test_unqualified_formal_inputs_fail_closed(design, change):
    solution, model, _, _ = design
    s = deepcopy(solution)
    s.update(change)
    with pytest.raises(ValueError):
        owner.forecast_precision_covariance(s, model=model, times_s=[1., 2., 3.])


def test_fit_epoch_and_invalid_joint_covariance_reject(design):
    solution, model, _, _ = design
    with pytest.raises(ValueError, match="follow every"):
        owner.forecast_precision_covariance(solution, model=model, times_s=[0., 2., 3.])
    with pytest.raises(ValueError):
        owner.forecast_precision_covariance(solution, model=model, times_s=[1., 2., 3.],
                                            process_noise_joint_covariance=np.eye(6))


@pytest.mark.parametrize("velocity", [False, True])
def test_precision_fit_correlated_covariance_row_order(monkeypatch, velocity):
    from sim.estimation.weighting import prepare_covariance_whitener, whiten_residual_with_factor
    from sim.pro_tracking_od import precision
    rng = np.random.default_rng(47)
    truth = rng.normal(size=(8, 6))
    observed = rng.normal(size=(8, 6))
    dimension = 48 if velocity else 24
    a = rng.normal(size=(dimension, dimension))
    r = a @ a.T + np.eye(dimension)
    class Predictor:
        def __init__(self, model, times):
            self.times = np.asarray(times)
        def predict(self, *a, **k):
            return truth
    class Captured(Exception):
        pass
    def solve(parameters, residual, **kw):
        white = residual(parameters.initial_native())
        raw = (truth - observed)[:, :6 if velocity else 3].ravel()
        expected = np.linalg.solve(np.linalg.cholesky(r), raw)
        np.testing.assert_allclose(white, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(white @ white, raw @ np.linalg.solve(r, raw), rtol=1e-12)
        raise Captured
    monkeypatch.setattr(precision, "PrecisionOrbitPredictor", Predictor)
    monkeypatch.setattr(precision, "solve_batch_least_squares", solve)
    kwargs = dict(model=SimpleNamespace(radiation_model="ecom2"), times_s=np.arange(8.),
                  positions_eme2000_km=observed[:, :3], initial_state=np.zeros(6),
                  velocities_eme2000_km_s=observed[:, 3:] if velocity else None)
    with pytest.raises(Captured):
        precision.fit_precision_orbit(**kwargs, fit_observation_covariance=r)
    with pytest.raises(ValueError, match="positive definite"):
        precision.fit_precision_orbit(**kwargs, fit_observation_covariance=np.ones_like(r))
    factor = prepare_covariance_whitener(r, dimension=dimension)
    np.testing.assert_allclose(whiten_residual_with_factor(np.ones(dimension), factor),
                               np.linalg.solve(factor, np.ones(dimension)), atol=1e-12)


def test_slr_local_model_preserves_precise_epoch_and_maintained_owners(monkeypatch):
    from sim.pro_tracking_od import precision_slr as slr
    calls=[]
    state=np.arange(6,dtype=float)
    def propagate(x,dt,t,*args):
        calls.append((x.copy(),dt,t))
        return x
    predictor=SimpleNamespace(model=SimpleNamespace(epoch_jd_utc=2460000.5,eop_path="pinned-eop"),
        propagator=SimpleNamespace(propagate=propagate),environment={})
    def geometry(**kw):
        assert kw["frame_model"]=="hpop_like" and kw["eop_path"]=="pinned-eop"
        kw["state_at_offset_s"](.1)
        return {"predicted_two_way_range_km":12.}
    monkeypatch.setattr(slr,"predict_slr_two_way_time_of_flight_s",geometry)
    monkeypatch.setattr(slr,"two_way_optical_media_correction",lambda *a,**k:{"one_way_equivalent_km":.002})
    row=dict(time_jd_utc=2460001.5,epoch_event=2,epoch_offset_s=2e-6,station_metadata={"ecef_km":[6000.,0.,0.]})
    assert slr.precision_slr_range_km(row,state,predictor=predictor,context=object())==12.002
    assert calls[0][1:]==pytest.approx((.100002,86400.),abs=1e-14)
    np.testing.assert_array_equal(calls[0][0],state)


def test_precision_slr_covariance_forwards_fitted_earth_radiation(monkeypatch):
    from sim.pro_tracking_od import precision_slr as slr

    calls = []
    epoch = 2460000.5

    class Model:
        earth_radiation = True
        epoch_jd_utc = epoch
        radiation_model = "cannonball"
        mu_km3_s2 = 398600.4418
        eop_path = "pinned-eop"

    class Predictor:
        def __init__(self, model, times):
            self.model = model
            self.times = np.asarray(times)
            self.propagator = SimpleNamespace(propagate=lambda state, *_args: np.asarray(state, dtype=float))
            self.environment = {}

        def predict(self, state, srp, *, ecom_coefficients_km_s2=None,
                    earth_radiation_cr_area_mass_m2_kg=None):
            calls.append((float(srp), ecom_coefficients_km_s2, earth_radiation_cr_area_mass_m2_kg))
            if earth_radiation_cr_area_mass_m2_kg is None:
                raise AssertionError("Earth-radiation coefficient was not forwarded")
            return np.tile(np.asarray(state, dtype=float), (len(self.times), 1))

    projected = {}

    def project(forecast, *, measurement_jacobians, observation_covariance, epoch_indices):
        projected.update(
            jacobians=np.asarray(measurement_jacobians),
            observation_covariance=np.asarray(observation_covariance),
            epoch_indices=np.asarray(epoch_indices),
        )
        return {"covariance": [[1.0]]}

    monkeypatch.setattr(slr, "PrecisionOrbitPredictor", Predictor)
    monkeypatch.setattr(
        slr,
        "forecast_precision_covariance",
        lambda solution, *, model, times_s, process_noise_joint_covariance: {
            "records": [{"state_eci_km_km_s": [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]}],
        },
    )
    monkeypatch.setattr(slr, "precision_slr_range_km", lambda row, state, **_kwargs: float(state[0]))
    monkeypatch.setattr(slr, "project_precision_covariance", project)

    row = {
        "time_jd_utc": epoch + 10.0 / 86400.0,
        "epoch_offset_s": 0.0,
        "epoch_event": 2,
        "station_metadata": {"ecef_km": [6378.137, 0.0, 0.0]},
        "measurement_id": "slr:0",
        "observed_two_way_range_km": 7000.0,
    }
    solution = {
        "initial_state_eme2000_km_km_s": [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0],
        "srp_cr_area_mass_m2_kg": 0.12,
        "earth_radiation_cr_area_mass_m2_kg": 0.037,
    }
    result = slr.precision_slr_covariance(
        solution, model=Model(), rows=[row], observation_covariance=np.eye(1)
    )

    assert calls == [(0.12, None, 0.037)]
    np.testing.assert_array_equal(projected["epoch_indices"], [0])
    np.testing.assert_array_equal(projected["observation_covariance"], np.eye(1))
    assert result["predicted_range_km"] == [7000.0]
