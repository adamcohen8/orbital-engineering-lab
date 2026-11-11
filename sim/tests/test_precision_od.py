"""Precision ONP numerical recovery, exact epochs and independent-fit boundaries."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sim.pro_tracking_od import precision as owner
from sim.tracking_od import PrecisionOrbitModel, PrecisionOrbitPredictor, fit_precision_orbit, predict_precision_orbit


@pytest.fixture
def model(tmp_path):
    eop = tmp_path / "earth.txt"
    eop.write_text("NUM_OBSERVED_POINTS 3\n" + "".join(
        f"0 0 0 {mjd} 0 0 0 0 0 0 0 0 37\n" for mjd in [60360, 60370, 60400]))
    gravity = tmp_path / "gravity.txt"
    gravity.write_text("2 0 -0.00048 0.0\n2 2 0.000002 -0.000001\n")
    return PrecisionOrbitModel(2460364.5, str(gravity), str(eop), gravity_degree=2, gravity_order=2)


def test_native_exact_schedule_recovery_and_prediction_never_refits(model, monkeypatch):
    seed = np.array([28000., 0., 2000., 0., 2.3, 2.8])
    times = np.r_[0., np.arange(900., 21601., 900.)]
    truth = PrecisionOrbitPredictor(model, times).predict(seed, .027)
    result = fit_precision_orbit(model=model, times_s=times, positions_eme2000_km=truth[:, :3],
        velocities_eme2000_km_s=truth[:, 3:], initial_state=seed + [.01, -.01, .01, 1e-5, -1e-5, 1e-5],
        initial_srp_area_mass_m2_kg=.025, max_nfev=30)
    assert result["solver"]["success"] and result["position_rms_m"] < .01
    assert abs(result["srp_cr_area_mass_m2_kg"] - .027) < 1e-5
    assert not result["empirically_calibrated"] and not result["rejected_observations"]
    monkeypatch.setattr(owner, "solve_batch_least_squares", lambda *a, **k: pytest.fail("Forecast refitted"))
    future = np.array([22000.125, 24123.456])
    expected = PrecisionOrbitPredictor(model, future).predict(seed, .027)
    actual = predict_precision_orbit(result, model=model, times_s=future)
    assert np.max(np.linalg.norm(actual[:, :3] - expected[:, :3], axis=1)) < 1e-5
    with pytest.raises(ValueError, match="follow every"):
        predict_precision_orbit(result, model=model, times_s=[times[-1]])


@pytest.mark.parametrize("times", [[-1], [0, 0], [1, 0], [np.nan], [[1]], [], [33*86400]])
def test_invalid_prediction_epochs_fail_closed(model, times):
    with pytest.raises(ValueError, match="epochs"):
        PrecisionOrbitPredictor(model, times)


def test_resource_drift_and_native_failure_reject(model, monkeypatch):
    predictor = PrecisionOrbitPredictor(model, [0., 91.125, 500.])
    assert predictor.starts[0] == 0. and predictor.widths[predictor.ends[1]-1] == 1.125
    Path(model.eop_path).write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        predictor.predict([28000, 0, 2000, 0, 2.3, 2.8], .026)


def test_missing_native_is_not_silently_approximated(model, monkeypatch):
    predictor = PrecisionOrbitPredictor(model, [60.])
    monkeypatch.setattr(predictor.propagator, "try_propagate_sampled_steps", lambda *a, **k: None)
    with pytest.raises(ValueError, match="requires the supported native"):
        predictor.predict([28000, 0, 2000, 0, 2.3, 2.8], .026)


@pytest.mark.parametrize("changes", [dict(step_s=0.), dict(step_s=True), dict(step_s=90.),
    dict(gravity_degree=21), dict(gravity_order=21), dict(gravity_degree=2.5), dict(epoch_jd_utc=np.nan)])
def test_invalid_models_reject(model, changes):
    with pytest.raises(ValueError):
        replace(model, **changes)


def test_position_only_fit_and_invalid_data(model):
    seed = [28000., 0., 2000., 0., 2.3, 2.8]
    times = np.arange(0., 43201., 1800.)
    truth = PrecisionOrbitPredictor(model, times).predict(seed, .026)
    result = fit_precision_orbit(model=model, times_s=times, positions_eme2000_km=truth[:, :3],
        initial_state=seed, max_nfev=5)
    assert result["solver"]["success"] and not result["velocities_used"]
    with pytest.raises(ValueError, match="sigmas"):
        fit_precision_orbit(model=model, times_s=times, positions_eme2000_km=truth[:, :3],
            initial_state=seed, position_sigma_km=0)


def test_ecom2_recovery_fixed_forecast_and_exclusive_radiation(model, monkeypatch):
    m = replace(model, radiation_model='ecom2', relativity=True, solid_earth_tides=True)
    seed = np.array([26000.,1000.,1000.,-.2,2.7,2.4])
    coefficients = [-1.2e-10,2e-12,-3e-12,4e-12,5e-12,-6e-12,7e-12]
    times = np.arange(0.,86401.,1800.)
    engine = PrecisionOrbitPredictor(m,times)
    truth = engine.predict(seed,0.,ecom_coefficients_km_s2=coefficients)
    fit = fit_precision_orbit(model=m,times_s=times,positions_eme2000_km=truth[:,:3],
        velocities_eme2000_km_s=truth[:,3:],initial_state=seed+[.01,-.01,.01,1e-5,-1e-5,1e-5],max_nfev=30)
    assert fit['solver']['success'] and fit['position_rms_m']<.005
    assert fit['diagnostics']['observability']['data_full_rank']
    np.testing.assert_allclose(fit['ecom2_coefficients_km_s2'],coefficients,atol=3e-14,rtol=0)
    monkeypatch.setattr(owner,'solve_batch_least_squares',lambda *a,**k: pytest.fail('forecast refit'))
    future=[90000.125,96000.75]
    expected=PrecisionOrbitPredictor(m,future).predict(seed,0.,ecom_coefficients_km_s2=coefficients)
    actual=predict_precision_orbit(fit,model=m,times_s=future)
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
    with pytest.raises(ValueError,match='replaces cannonball'):
        engine.predict(seed,.02,ecom_coefficients_km_s2=coefficients)
    with pytest.raises(ValueError,match='cannot accept'):
        PrecisionOrbitPredictor(model,times).predict(seed,.02,ecom_coefficients_km_s2=coefficients)
