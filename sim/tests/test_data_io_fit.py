from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.data_io.catalog import propagate_catalog, read_catalog
from sim.data_io.contracts import DataIOError
from sim.data_io.fit import fit_review_catalog
from sim.data_io.trajectory import transform_trajectory
from sim.tests.data_io_helpers import make_review

ROOT = Path(__file__).resolve().parents[2]


def template():
    return read_catalog(ROOT / "sim/interchange/examples/oel_sgp4_mean_elements_v3.omm", format="omm-kvn")[0]


def test_fit_review_invokes_existing_estimator_and_checks_encoded_tle_on_holdout(tmp_path):
    prior = template()
    history = propagate_catalog(prior, start_epoch="2024-01-01T00:00:00", duration_s=1200, step_s=60)
    inertial = transform_trajectory(history, "EME2000")
    run = make_review(tmp_path / "run", times=inertial.times_s, states=inertial.states)
    result = fit_review_catalog(
        run,
        template=prior,
        output_dir=tmp_path / "fit",
        fit_until_s=840,
        max_position_rms_m=20,
        max_velocity_rms_m_s=0.02,
        formats=("omm-kvn", "omm-xml", "tle", "gp-json", "gp-csv"),
    )
    assert result["status"] == "accepted"
    evidence = json.loads(Path(result["evidence_path"]).read_text())
    assert evidence["simulation_executed"] is False
    assert evidence["source"]["source_run_id"] == "io-test-run"
    assert evidence["encoded_product_metrics"]["tle"]["holdout"]["samples"] == 6
    assert evidence["estimator"]["holdout_sample_count"] == 6
    assert len(result["outputs"]) == 5


def test_encoded_catalog_rejection_retains_evidence_without_promoting_files(tmp_path, monkeypatch):
    import sim.data_io.fit as owner

    prior = template()
    history = transform_trajectory(
        propagate_catalog(prior, start_epoch="2024-01-01T00:00:00", duration_s=1200, step_s=60), "EME2000"
    )
    run = make_review(tmp_path / "run", times=history.times_s, states=history.states)
    original = owner.encode_catalog

    def corrupt_serialization(records, *, format):
        data = json.loads(original(records, format=format))
        data[0]["MEAN_ANOMALY"] = (float(data[0]["MEAN_ANOMALY"]) + 0.1) % 360
        return json.dumps(data).encode()

    monkeypatch.setattr(owner, "encode_catalog", corrupt_serialization)
    result = fit_review_catalog(
        run,
        template=prior,
        output_dir=tmp_path / "rejected",
        fit_until_s=840,
        max_position_rms_m=20,
        max_velocity_rms_m_s=0.02,
        formats=("gp-json",),
    )
    assert result["status"] == "rejected" and result["outputs"] == []
    evidence = json.loads(Path(result["evidence_path"]).read_text())
    assert evidence["encoded_product_metrics"]["gp-json"]["holdout"]["position_rms_m"] > 20
    assert sorted(p.name for p in (tmp_path / "rejected").iterdir()) == [
        "fit-evidence.json",
        "fit-evidence.json.receipt.json",
    ]


def test_fit_review_refuses_missing_holdout(tmp_path):
    prior = template()

    run = make_review(tmp_path / "run")
    with pytest.raises(DataIOError, match="holdout"):
        fit_review_catalog(
            run,
            template=prior,
            output_dir=tmp_path / "fit",
            fit_until_s=60,
            max_position_rms_m=1,
            max_velocity_rms_m_s=1,
        )
    assert not (tmp_path / "fit").exists()


def test_real_onp_run_is_fitted_only_after_review_completion(tmp_path):
    from sim.data_io.materialize import materialize_scenario
    from sim.execution import run_simulation_config_file

    prior = template()
    inertial = transform_trajectory(
        propagate_catalog(prior, start_epoch="2024-01-01T00:00:00", duration_s=60, step_s=30), "EME2000"
    )
    scenario = tmp_path / "onp.yaml"
    materialized = materialize_scenario(
        inertial, scenario, scenario_name="onp_review_fit", output_dir=tmp_path / "onp", duration_s=1200, step_s=30
    )
    assert materialized["validation"]["ok"]
    from sim.security.config_paths import ConfigPathPolicy

    run_simulation_config_file(
        scenario, path_policy=ConfigPathPolicy.default(config_path=scenario, write_roots=(tmp_path,))
    )
    result = fit_review_catalog(
        tmp_path / "onp",
        template=prior,
        output_dir=tmp_path / "derived",
        fit_until_s=840,
        max_position_rms_m=5000,
        max_velocity_rms_m_s=10,
        formats=("omm-kvn", "tle"),
    )
    assert result["status"] == "accepted"
    report = json.loads(Path(result["evidence_path"]).read_text())
    assert report["source"]["interpretation"].startswith("completed simulated truth")
    assert report["encoded_product_metrics"]["tle"]["holdout"]["samples"] == 12
