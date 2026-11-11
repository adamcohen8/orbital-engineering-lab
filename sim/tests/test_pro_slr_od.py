from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sim.pro_slr_od import (
    ProSlrOdError,
    ProSlrOdProblem,
    replay_evidence,
    run_pro_slr_od,
    validate_problem,
)
from sim.pro_slr_od.__main__ import main
from sim.pro_slr_od.core import _measurement_rows, _partition
from sim.slr_tracking import parse_ilrs_crd_v2_normal_points
from sim.utils.io import SafeReadError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples/tracking_od/synthetic_problem.json"
CRD = REPO_ROOT / "examples/tracking_od/synthetic_crd_v2.np2"


def test_pro_slr_od_validates_runs_and_replays(tmp_path: Path) -> None:
    problem = ProSlrOdProblem.from_json(EXAMPLE)
    assert problem.sigma_floor_ps == pytest.approx(50.0)
    assert problem.to_dict()["measurement"] == {"sigma_floor_ps": 50.0}
    validation = validate_problem(problem)
    assert validation["fit_count"] == 8
    assert validation["validation_count"] == 3
    assert validation["holdout_count"] == 3
    # The frozen example's 40-evaluation cap is an input contract. Test
    # convergence with an explicit independent budget: SciPy/BLAS builds can
    # need different iteration counts on this weakly observable short arc.
    assert problem.batch_max_nfev == 40
    problem = replace(problem, batch_max_nfev=80)
    output = tmp_path / "evidence"
    result = run_pro_slr_od(problem, output)
    assert result["status"] == "complete"
    assert result["batch_summary"]["solver"]["nfev"] <= 80
    assert result["batch_summary"]["solver"]["success"] is True
    assert result["sequential_summary"]["covariance_checks"]["all_symmetric_psd"] is True
    assert result["validation_statistics"]["finite"] is True
    assert result["holdout_statistics"]["finite"] is True
    assert result["claim_posture"]["calibrated_covariance"] is False
    assert replay_evidence(output)["status"] == "verified"


def test_frozen_example_evaluation_budget_is_honored(tmp_path: Path) -> None:
    problem = ProSlrOdProblem.from_json(EXAMPLE)
    result = run_pro_slr_od(problem, tmp_path / "bounded-evidence")
    solver = result["batch_summary"]["solver"]
    assert problem.batch_max_nfev == 40
    assert solver["nfev"] <= problem.batch_max_nfev
    if not solver["success"]:
        assert result["batch_summary"]["quality_gates"]["evidence_status"] == "review_required"


def test_contract_evidence_and_external_file_reads_fail_closed(tmp_path: Path) -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["typo"] = True
    with pytest.raises(ProSlrOdError, match="Unknown problem"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)

    output = tmp_path / "evidence"
    run_pro_slr_od(ProSlrOdProblem.from_json(EXAMPLE), output)
    summary = output / "pro_slr_od_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "tampered"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProSlrOdError, match="receipt mismatch"):
        replay_evidence(output)

    link = tmp_path / "linked-crd.np2"
    link.symlink_to(CRD)
    with pytest.raises(SafeReadError):
        parse_ilrs_crd_v2_normal_points(link)
    link.unlink()


def test_relative_output_path_replays_without_atomic_directory_drift(monkeypatch, tmp_path: Path) -> None:
    problem = ProSlrOdProblem.from_json(EXAMPLE)
    monkeypatch.chdir(tmp_path)
    run_pro_slr_od(problem, Path("relative-evidence"))
    assert replay_evidence(Path("relative-evidence"))["status"] == "verified"


def test_replay_rejects_self_receipted_prediction_and_manifest_claims(tmp_path: Path) -> None:
    problem = ProSlrOdProblem.from_json(EXAMPLE)
    forged = tmp_path / "forged-artifact"
    run_pro_slr_od(problem, forged)
    prediction = forged / "validation_predictions.csv"
    prediction.write_bytes(prediction.read_bytes() + b"FORGED\n")
    manifest_path = forged / "pro_slr_od_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = next(item for item in manifest["files"] if item["path"] == prediction.name)
    content = prediction.read_bytes()
    receipt.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProSlrOdError, match="every retained artifact"):
        replay_evidence(forged)

    contradicted = tmp_path / "contradicted-manifest"
    run_pro_slr_od(problem, contradicted)
    manifest_path = contradicted / "pro_slr_od_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest["problem_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ProSlrOdError, match="manifest claims"):
        replay_evidence(contradicted)


def test_partition_requires_distinct_validation_and_untouched_holdout() -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["windows"]["fit_duration_s"] = 1200.0
    raw["windows"]["validation_duration_s"] = 50.0
    raw["windows"]["holdout_duration_s"] = 50.0
    with pytest.raises(ProSlrOdError, match="validation row"):
        validate_problem(ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent))


def test_slr_boundary_tolerance_does_not_pull_post_boundary_into_fit() -> None:
    problem = ProSlrOdProblem.from_json(EXAMPLE)
    rows = _measurement_rows(problem)
    rows[8]["time_s"] = problem.fit_duration_s + 0.0005
    fit, validation, holdout = _partition(rows, problem)
    assert len(fit) == 8
    assert len(validation) == 3
    assert len(holdout) == 3
    assert rows[8] in validation


def test_measurement_sigma_floor_is_explicit_and_positive() -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["measurement"] = {"sigma_floor_ps": 6671.281904}
    problem = ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)
    assert problem.sigma_floor_ps == pytest.approx(6671.281904)
    raw["measurement"]["sigma_floor_ps"] = 0.0
    with pytest.raises(ProSlrOdError, match="must be positive"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)


def test_process_noise_contract_errors_are_wrapped() -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["sequential"]["spectral_density"] = "not-a-vector"
    with pytest.raises(ProSlrOdError, match="process noise is invalid"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)


def test_pro_slr_od_rejects_ambiguous_station_and_solver_inputs() -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    first_code, first_station = next(iter(raw["station_catalog"].items()))
    raw["station_catalog"][first_code.lower()] = dict(first_station)
    with pytest.raises(ProSlrOdError, match="duplicate station code"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)

    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["batch"]["max_nfev"] = 120.5
    with pytest.raises(ProSlrOdError, match="positive integer"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)

    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["crd_text"] = None
    raw.pop("crd_path", None)
    with pytest.raises(ProSlrOdError, match="UTF-8 string"):
        ProSlrOdProblem.from_mapping(raw, base_dir=EXAMPLE.parent)


def test_cli_validate(tmp_path: Path, capsys) -> None:
    assert main(["validate", str(EXAMPLE)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
