import hashlib
import json
from pathlib import Path

import pytest
import yaml

from sim.analysis.access_chains import Link, evaluate_chain, parse_chain
from sim.api import SimulationConfig, SimulationSession
from sim.review.__main__ import main as review_main
from sim.review.access_chains import evaluate_completed_run
from sim.review.workspace import ReviewWorkspace


def _window(index, start, end, *, disposition="provider_refined"):
    return {
        "interval_index": index, "start_s": start, "end_s": end,
        "start_censored": start == 0, "end_censored": end == 60,
        "acquisition_disposition": "study_start_censored" if start == 0 else disposition,
        "loss_disposition": "study_end_censored" if end == 60 else disposition,
        "acquisition_reason": "below_threshold", "loss_reason": "below_threshold",
    }


def test_access_chains_known_answer():
    imager, a, b = [("spacecraft", key) for key in ("imager", "a", "b")]
    denver, boulder = [("fixed_wgs84_site", key) for key in ("denver", "boulder")]
    catalog = {
        "imager_a": Link(imager, a, 1_000_000, "a.rx", "imager.tx"),
        "imager_b": Link(imager, b, 1_000_000, "b.rx", "imager.tx"),
        "a_denver": Link(a, denver, 1_000_000, "denver.rx", "a.tx"),
        "b_boulder": Link(b, boulder, 1_000_000, "boulder.rx", "b.tx"),
        "a_touch": Link(a, denver, 1_000_000, "denver.rx", "a.tx"),
    }
    spec = {
        "analysis_id": "live_downlink", "service_rate_bps": 1_000_000,
        "simultaneous_relays": ["a", "b"],
        "paths": [
            {"path_id": "a", "links": ["imager_a", "a_denver"]},
            {"path_id": "b", "links": ["imager_b", "b_boulder"]},
            {"path_id": "touch", "links": ["imager_a", "a_touch"]},
        ],
        "sensor_gate": {"coverage_analysis_id": "imaging", "cell_index": 8},
    }
    coverage = {"imaging": ("imager", 5)}
    chain = parse_chain(spec, catalog, coverage)
    times = [0, 10, 20, 25, 30, 31, 32, 38, 40, 45, 55, 60]
    data = {}
    for key, (start, stop) in {
        "imager_a": (10, 40), "a_denver": (25, 55), "imager_b": (30, 45),
        "b_boulder": (10, 60), "a_touch": (40, 60),
    }.items():
        data[key] = {
            "windows": [_window(0, start, stop)],
            "samples": [
                {"time_s": t, "available": start <= t < stop,
                 "margin_db": 5 if key.startswith("imager") else (4 if (key == "a_denver") == (t != 31) else 3)}
                for t in times
            ],
            "input_evidence_sha256": "fixture", "semantic_sha256": "fixture",
        }
    sensor = {
        "times_s": times,
        "available": [t < 32 or 38 <= t < 55 for t in times],
        "windows": [_window(0, 0, 32, disposition="sample_bounded"), _window(1, 38, 55)],
        "provenance": {"target_kind": "healpix_cell_center"},
    }
    result = evaluate_chain(chain, catalog, data, sensor=sensor)
    assert result["service_windows"] == [{"start_s": 25, "end_s": 32}, {"start_s": 38, "end_s": 45}]
    assert result["summary"]["available_duration_s"] == 14
    assert result["summary"]["redundant_duration_s"] == 4
    assert result["paths"][2]["windows"] == []
    assert result["paths"][0]["windows"][0]["end_evidence"] == [{
        "analysis_id": "sensor:imaging", "interval_index": 0, "censored": False,
        "disposition": "sample_bounded", "reason": "below_threshold",
    }]
    ranks = {row["time_s"]: row["path_id"] for row in result["ranked_samples"]}
    assert (ranks[30], ranks[31]) == ("a", "b")
    outage = evaluate_chain(chain, catalog, data, sensor=sensor, exclude_ground_stations=["denver"])
    assert outage["lost_service_segments"] == [{"start_s": 25, "end_s": 30, "path_ids": ["a"]}]
    assert outage["summary"]["lost_service_duration_s"] == 5
    assert outage["summary"]["available_duration_s"] == 9

    discovered_spec = {key: value for key, value in spec.items() if key != "paths"}
    discovered_spec["discover"] = {
        "source_object_id": "imager", "destination_ground_station_ids": ["denver", "boulder"], "max_hops": 2,
    }
    discovered = parse_chain(discovered_spec, catalog, coverage)
    assert {route.links for route in discovered.paths} == {route.links for route in chain.paths}
    with pytest.raises(ValueError, match="simultaneous relay"):
        parse_chain({**spec, "simultaneous_relays": ["a"]}, catalog, coverage)
    with pytest.raises(ValueError, match="below service_rate"):
        parse_chain({**spec, "service_rate_bps": 2_000_000}, catalog, coverage)
    with pytest.raises(ValueError, match="disconnected"):
        parse_chain({**spec, "paths": [{"path_id": "bad", "links": ["a_denver", "imager_a"]}]}, catalog, coverage)
    with pytest.raises(ValueError, match="same retained epochs"):
        altered = {**data, "imager_a": {**data["imager_a"], "samples": data["imager_a"]["samples"][1:]}}
        evaluate_chain(chain, catalog, altered, sensor=sensor)


@pytest.mark.parametrize("target", [{"latitude_deg": 90.0, "longitude_deg": 0.0}, {"cell_index": 1023}])
def test_access_chain_completed_run_replay(tmp_path, target):
    example = Path(__file__).resolve().parents[2] / "examples/configs/public_access_chains.yaml"
    config = yaml.safe_load(example.read_text())
    output = config["outputs"]
    output["output_dir"] = str(tmp_path / "scenario")
    section = output["orbital_analysis"]
    spec = section.pop("access_chains")[0]
    spec["sensor_gate"].pop("latitude_deg")
    spec["sensor_gate"].pop("longitude_deg")
    spec["sensor_gate"].update(target)
    spec["include_timeline"] = False
    SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    db = Path(output["output_dir"]) / "review/run.sqlite"
    digest = hashlib.sha256(db.read_bytes()).digest()
    _, from_review = evaluate_completed_run(db, spec)
    assert from_review["paths"]
    assert from_review["summary"]["available_duration_s"] > 0
    assert "access_chains" not in section
    assert hashlib.sha256(db.read_bytes()).digest() == digest
    if "latitude_deg" in target:
        derived = tmp_path / "outage"
        assert review_main([
            "access-chains", str(db), "--request-json", json.dumps({**spec, "include_timeline": True}),
            "--exclude-ground-station", "north_site", "--output-dir", str(derived),
        ]) == 0
        assert json.loads((derived / "access_chain.json").read_text())["summary"]["lost_service_duration_s"] > 0
        assert (derived / "access_chain_review.html").is_file()
        assert (derived / "access_chain_timeline.quality.json").is_file()

    section["access_chains"] = [spec]
    output["output_dir"] = str(tmp_path / "with_chain")
    result = SimulationSession.from_config(SimulationConfig.from_dict(config)).run()
    stored = json.loads(Path(result.artifacts["orbital_analysis"]["access_chains"][0]["evidence_json"]).read_text())
    from_review.pop("source_run")
    assert stored == json.loads(json.dumps(from_review))
    with ReviewWorkspace.open(output["output_dir"]) as workspace:
        assert workspace.query("SELECT available_duration_s FROM access_chain_summary").rows == [{
            "available_duration_s": stored["summary"]["available_duration_s"]
        }]
        assert workspace.query("SELECT * FROM access_chain_windows").row_count
