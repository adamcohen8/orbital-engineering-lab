import json

import pytest

from sim.review.__main__ import main
from sim.review.observation_comparison import SCHEMA, compare_observations

BASE = {"schema": SCHEMA, "source": "fixture", "id_namespace": "survey-1"}


def test_unknown_published_mask_against_named_rows():
    rows = [{"observation_id": str(i), "partition": "fit" if i < 561 else "excluded"} for i in range(569)]
    report = compare_observations({**BASE, "counts": {"fit": 478}},
                                  {**BASE, "counts": {"available": 569}, "observations": rows})
    assert report["identity_status"] == "unknown_aggregate_only"
    assert report["count_delta_right_minus_left"] == {"fit": 83}
    assert report["fit_overlap"] is report["treatment_comparison"] is None
    assert report["left"]["ids_by_partition"] is None
    assert len(report["right"]["ids_by_partition"]["excluded"]) == 8


def test_treatments_partitions_and_unknowns():
    row = {"observation_id": "a", "partition": "fit", "weight": {"robust_factor": 0.8},
           "catalog_correction": {"ra_arcsec": 0}, "uncertainty": {"ra_arcsec": 0.5}}
    left = {**BASE, "observations": [row, {"observation_id": "b", "partition": "fit"}]}
    right = {**BASE, "observations": [{**row, "partition": "holdout", "weight": {"robust_factor": 1.0},
             "catalog_correction": None, "uncertainty": {"ra_arcsec": 1, "dec_arcsec": 0.3}}]}
    report = compare_observations(left, right)
    assert report["partition_changes"] == [{"observation_id": "a", "left": "fit", "right": "holdout"}]
    assert report["fit_overlap"] == [] and report["only_left"] == ["b"]
    treatments = report["treatment_comparison"]
    assert treatments["uncertainty"]["changed"][0]["fields"]["ra_arcsec"] == {"left": 0.5, "right": 1}
    assert treatments["uncertainty"]["unknown"][0]["fields"] == ["dec_arcsec"]
    assert treatments["catalog_correction"]["unknown"][0]["fields"] == ["ra_arcsec"]
    assert treatments["weight"]["changed"][0]["fields"]["robust_factor"]["right"] == 1
    assert compare_observations(left, {**right, "id_namespace": "other"})["fit_overlap"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        compare_observations(left, {**BASE, "observations": [row, row]})
    with pytest.raises(ValueError, match="disagree"):
        compare_observations(left, {**right, "counts": {"fit": 1}})


def test_native_residuals_via_review_cli(tmp_path, capsys):
    (tmp_path / "fit.csv").write_text("observation_id,partition,position_sigma_km\na,fit,0.05\n")
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({**BASE, "residual_csv": ["fit.csv"], "counts": {"excluded": 4}}))
    assert main(["compare-observations", str(path), str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fit_overlap"] == ["a"]
    assert report["left"]["ids_by_partition"]["excluded"] is None
    assert report["left"]["counts"]["excluded"] == 4
    assert report["treatment_comparison"]["uncertainty"]["matching_observation_count"] == 1
