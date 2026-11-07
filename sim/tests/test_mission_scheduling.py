from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sim.analysis.mission_scheduling import (
    MissionSchedulingError,
    MissionSchedulingProblem,
    replay_mission_schedule,
    solve_mission_schedule,
    verify_mission_scheduling_artifacts,
    write_mission_scheduling_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/mission_scheduling/public_two_asset_collection_problem.json"


def _payload() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _problem() -> MissionSchedulingProblem:
    return MissionSchedulingProblem.from_mapping(_payload())


def test_hand_case_selects_both_observations_and_noncontended_downlinks() -> None:
    result = solve_mission_schedule(_problem())

    assert result.status == "complete"
    assert result.objective_value == 19.0
    assert result.selected_opportunity_ids == ("A-DL-GS1", "A-OBS-1", "B-DL-GS2", "B-OBS-1")
    assert all(item.fully_delivered for item in result.deliveries)
    assert {item.opportunity_id: item.delivered_bytes for item in result.deliveries} == {
        "A-OBS-1": 100.0,
        "B-OBS-1": 90.0,
    }
    assert result.resource_summary["SAT-A"]["final_storage_bytes"] == 0.0
    assert result.resource_summary["SAT-B"]["final_storage_bytes"] == 0.0
    assert "station_contention" in result.rejected_opportunity_reasons["B-DL-GS1-CONTENDED"]


def test_input_order_does_not_change_semantic_result() -> None:
    payload = _payload()
    forward = solve_mission_schedule(payload)
    payload["assets"].reverse()
    payload["opportunities"].reverse()
    reversed_result = solve_mission_schedule(payload)

    assert reversed_result.selected_opportunity_ids == forward.selected_opportunity_ids
    assert reversed_result.input_semantic_sha256 == forward.input_semantic_sha256
    assert reversed_result.schedule_semantic_sha256 == forward.schedule_semantic_sha256


def test_exact_result_matches_independent_exhaustive_oracle() -> None:
    payload = _payload()
    candidates = payload["opportunities"]
    values: list[tuple[float, int, tuple[str, ...]]] = []
    for mask in range(1 << len(candidates)):
        chosen = [item for index, item in enumerate(candidates) if mask & (1 << index)]
        if sum(item["kind"] == "observation" for item in chosen) < 2:
            continue
        feasible = True
        for asset in ("SAT-A", "SAT-B"):
            tasks = sorted((item for item in chosen if item["asset_id"] == asset), key=lambda item: item["start_s"])
            if any(second["start_s"] < first["end_s"] + 2.0 for first, second in itertools.pairwise(tasks)):
                feasible = False
            produced = sum(item.get("data_volume_bytes", 0.0) for item in tasks)
            capacity = sum(item.get("downlink_capacity_bytes", 0.0) for item in tasks)
            for observation in (item for item in tasks if item["kind"] == "observation"):
                later_capacity = sum(
                    item.get("downlink_capacity_bytes", 0.0)
                    for item in tasks
                    if item["kind"] == "downlink" and item["start_s"] >= observation["end_s"]
                )
                if later_capacity < observation["data_volume_bytes"]:
                    feasible = False
            if produced > 150.0 or capacity < produced:
                feasible = False
        station_tasks = sorted(
            (item for item in chosen if item.get("station_id") == "GS-1"), key=lambda item: item["start_s"]
        )
        if any(second["start_s"] < first["end_s"] for first, second in itertools.pairwise(station_tasks)):
            feasible = False
        if feasible:
            ids = tuple(sorted(item["opportunity_id"] for item in chosen))
            values.append((sum(item["objective_value"] for item in chosen), -len(ids), ids))
    best_value = max(item[0] for item in values)
    best_count = min(-item[1] for item in values if item[0] == best_value)
    best_ids = min(
        item[2] for item in values if item[0] == best_value and -item[1] == best_count
    )

    result = solve_mission_schedule(payload)
    assert result.objective_value == best_value
    assert result.selected_opportunity_ids == best_ids


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda p: p["assets"][0].update(energy_budget_wh=2.0), "energy_budget"),
        (lambda p: p["assets"][0].update(storage_capacity_bytes=50.0), "storage_capacity"),
        (lambda p: p["assets"][0].update(maximum_payload_duty_cycle=0.01), "payload_duty_cycle"),
    ],
)
def test_resource_constraints_are_enforced(mutation, reason: str) -> None:
    payload = _payload()
    payload["minimum_selected_observations"] = 0
    mutation(payload)
    result = solve_mission_schedule(payload)

    assert result.status == "complete"
    assert any(reason in value for value in result.rejected_opportunity_reasons.values())


def test_observation_requires_a_later_downlink() -> None:
    payload = _payload()
    payload["opportunities"] = [payload["opportunities"][0], payload["opportunities"][1]]
    payload["opportunities"][1]["start_s"] = 0.0
    payload["opportunities"][1]["end_s"] = 5.0
    payload["minimum_selected_observations"] = 1

    result = solve_mission_schedule(payload)
    assert result.status == "infeasible"
    assert result.selected_opportunity_ids == ()


def test_direct_slew_plus_settling_is_enforced() -> None:
    payload = _payload()
    payload["minimum_selected_observations"] = 1
    payload["opportunities"] = [payload["opportunities"][0], payload["opportunities"][1]]
    payload["opportunities"][1]["start_s"] = 20.0
    payload["opportunities"][1]["end_s"] = 30.0
    payload["opportunities"][1]["pointing_unit_eci"] = [0.0, 1.0, 0.0]

    result = solve_mission_schedule(payload)
    assert result.status == "infeasible"


def test_station_contention_is_cross_asset() -> None:
    payload = _payload()
    payload["opportunities"] = payload["opportunities"][:4]
    result = solve_mission_schedule(payload)

    assert result.status == "infeasible"


def test_artifacts_and_authoritative_replay_are_content_bound(tmp_path: Path) -> None:
    result = solve_mission_schedule(_problem())
    artifacts = write_mission_scheduling_artifacts(result, tmp_path / "evidence")
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))

    replay = replay_mission_schedule(
        _problem(),
        selected_opportunity_ids=manifest["selected_opportunity_ids"],
        expected_input_semantic_sha256=manifest["input_semantic_sha256"],
        expected_schedule_semantic_sha256=manifest["schedule_semantic_sha256"],
        expected_status=manifest["status"],
    )
    assert replay["status"] == "verified"
    assert {item["path"] for item in manifest["artifacts"]} == {
        "mission_data_delivery.csv",
        "mission_resource_summary.csv",
        "mission_schedule.csv",
        "mission_schedule_rejections.csv",
        "mission_schedule_summary.json",
        "normalized_problem.json",
    }
    with pytest.raises(MissionSchedulingError, match="authoritative exact optimum"):
        replay_mission_schedule(
            _problem(),
            selected_opportunity_ids=("A-OBS-1", "A-DL-GS1"),
            expected_input_semantic_sha256=manifest["input_semantic_sha256"],
            expected_schedule_semantic_sha256=manifest["schedule_semantic_sha256"],
        )
    with pytest.raises(MissionSchedulingError, match="refusing to mix"):
        write_mission_scheduling_artifacts(result, artifacts.output_dir)

    manifest["objective_value"] = 20.0
    artifacts.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="manifest claims"):
        verify_mission_scheduling_artifacts(artifacts.output_dir)
    manifest["objective_value"] = 19.0
    artifacts.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    artifacts.schedule_csv.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="Artifact receipt mismatch"):
        verify_mission_scheduling_artifacts(artifacts.output_dir)


def test_cli_solve_and_replay(tmp_path: Path) -> None:
    output = tmp_path / "cli-evidence"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    solve = subprocess.run(
        [sys.executable, "-m", "sim.mission_scheduling", "solve", str(EXAMPLE), "--output-dir", str(output)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    replay = subprocess.run(
        [sys.executable, "-m", "sim.mission_scheduling", "replay", str(output)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert solve.returncode == 0, solve.stdout + solve.stderr
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)["status"] == "verified"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(maximum_candidates=19),
        lambda p: p["opportunities"][0].update(source_product_sha256="bad"),
        lambda p: p["opportunities"][0].update(asset_id="UNKNOWN"),
        lambda p: p["opportunities"][1].update(objective_value=1.0),
        lambda p: p["opportunities"][0].update(pointing_unit_eci=None),
    ],
)
def test_invalid_problems_fail_closed(mutation) -> None:
    payload = _payload()
    mutation(payload)
    with pytest.raises(MissionSchedulingError):
        MissionSchedulingProblem.from_mapping(payload)


@pytest.mark.parametrize("field", ["maximum_candidates", "minimum_selected_observations"])
def test_integer_controls_reject_float_coercion(field: str) -> None:
    payload = _payload()
    payload[field] = 2.0
    with pytest.raises(MissionSchedulingError, match="must be an integer"):
        MissionSchedulingProblem.from_mapping(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(surprise=True),
        lambda p: p["assets"][0].update(max_payload_duty_cycle=0.01),
        lambda p: p["opportunities"][0].update(surprise=True),
    ],
)
def test_unknown_problem_asset_and_opportunity_fields_fail_closed(mutation) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(MissionSchedulingError, match="unknown fields"):
        MissionSchedulingProblem.from_mapping(payload)


def test_strict_objective_ordering_never_prefers_a_lower_value() -> None:
    payload = _payload()
    first = payload["opportunities"][0]
    second = {**first, "opportunity_id": "B-OBS", "objective_value": first["objective_value"] + 5e-13}
    payload["opportunities"] = [first, second]
    payload["require_observation_delivery_by_horizon"] = False
    payload["minimum_selected_observations"] = 1

    result = solve_mission_schedule(payload)

    assert result.selected_opportunity_ids == ("B-OBS",)
    assert result.objective_value == second["objective_value"]


@pytest.mark.parametrize("pointing", [1, "1,0,0", [[1.0, 0.0, 0.0]]])
def test_malformed_pointing_is_a_structured_validation_error(pointing) -> None:
    payload = _payload()
    payload["opportunities"][0]["pointing_unit_eci"] = pointing

    with pytest.raises(MissionSchedulingError, match="pointing_unit_eci"):
        MissionSchedulingProblem.from_mapping(payload)


def test_cli_reports_malformed_pointing_without_a_traceback(tmp_path: Path) -> None:
    payload = _payload()
    payload["opportunities"][0]["pointing_unit_eci"] = 1
    problem = tmp_path / "bad-pointing.json"
    problem.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sim.mission_scheduling",
            "solve",
            str(problem),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "error"


def test_nonfinite_horizon_aggregate_and_oversized_asset_inventory_fail_closed() -> None:
    horizon = _payload()
    horizon["horizon_start_s"] = -1e308
    horizon["horizon_end_s"] = 1e308
    with pytest.raises(MissionSchedulingError, match="horizon duration must be finite"):
        MissionSchedulingProblem.from_mapping(horizon)

    aggregate = _payload()
    first = {**aggregate["opportunities"][0], "objective_value": 1e308}
    second = {
        **first,
        "opportunity_id": "SECOND-OBS",
        "start_s": 60.0,
        "end_s": 70.0,
        "objective_value": 1e308,
    }
    aggregate["opportunities"] = [first, second]
    aggregate["require_observation_delivery_by_horizon"] = False
    with pytest.raises(MissionSchedulingError, match="finite aggregate"):
        MissionSchedulingProblem.from_mapping(aggregate)

    inventory = _payload()
    inventory["assets"] = [
        {**inventory["assets"][0], "asset_id": f"SAT-{index}"} for index in range(19)
    ]
    with pytest.raises(MissionSchedulingError, match="bounded public inventory"):
        MissionSchedulingProblem.from_mapping(inventory)

    text = _payload()
    text["analysis_id"] = "x" * 257
    with pytest.raises(MissionSchedulingError, match="character public bound"):
        MissionSchedulingProblem.from_mapping(text)


def test_replay_rejects_manifest_receipt_inventory_and_size_drift(tmp_path: Path) -> None:
    first = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "manifest-inventory"
    )
    manifest = json.loads(first.manifest_json.read_text(encoding="utf-8"))
    manifest["surprise"] = True
    first.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="manifest field inventory"):
        verify_mission_scheduling_artifacts(first.output_dir)

    second = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "receipt-inventory"
    )
    manifest = json.loads(second.manifest_json.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["surprise"] = True
    second.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="receipt field inventory"):
        verify_mission_scheduling_artifacts(second.output_dir)

    third = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "receipt-size"
    )
    manifest = json.loads(third.manifest_json.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["bytes"] = 4 * 1024 * 1024 + 1
    third.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="public size bound"):
        verify_mission_scheduling_artifacts(third.output_dir)

    fourth = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "directory-inventory"
    )
    (fourth.output_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="directory inventory"):
        verify_mission_scheduling_artifacts(fourth.output_dir)

    fifth = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "nonfinite-manifest"
    )
    manifest = json.loads(fifth.manifest_json.read_text(encoding="utf-8"))
    manifest["objective_value"] = float("nan")
    fifth.manifest_json.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissionSchedulingError, match="forbidden non-finite"):
        verify_mission_scheduling_artifacts(fifth.output_dir)


@pytest.mark.parametrize(
    "artifact_name",
    [
        "mission_schedule_summary.json",
        "mission_schedule.csv",
        "mission_schedule_rejections.csv",
        "mission_resource_summary.csv",
        "mission_data_delivery.csv",
    ],
)
def test_replay_semantically_rejects_forged_derived_artifacts_with_updated_receipts(
    tmp_path: Path, artifact_name: str
) -> None:
    artifacts = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "evidence"
    )
    forged = b"forged\n"
    (artifacts.output_dir / artifact_name).write_bytes(forged)
    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    receipt = next(item for item in manifest["artifacts"] if item["path"] == artifact_name)
    receipt["bytes"] = len(forged)
    receipt["sha256"] = hashlib.sha256(forged).hexdigest()
    artifacts.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(MissionSchedulingError, match="authoritative deterministic replay"):
        verify_mission_scheduling_artifacts(artifacts.output_dir)


def test_verified_replay_returns_authoritative_activity_records(tmp_path: Path) -> None:
    artifacts = write_mission_scheduling_artifacts(
        solve_mission_schedule(_problem()), tmp_path / "evidence"
    )

    verified = verify_mission_scheduling_artifacts(artifacts.output_dir)

    assert [item["opportunity_id"] for item in verified["activities"]] == [
        item.opportunity_id for item in solve_mission_schedule(_problem()).activities
    ]
    assert verified["resource_summary"]["SAT-A"]["final_storage_bytes"] == 0.0


def test_artifact_publication_refuses_existing_directories_and_is_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = solve_mission_schedule(_problem())
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(MissionSchedulingError, match="refusing to mix"):
        write_mission_scheduling_artifacts(result, existing)
    assert list(existing.iterdir()) == []

    parent = tmp_path / "atomic"
    parent.mkdir()
    destination = parent / "evidence"
    original_write_bytes = Path.write_bytes

    def fail_during_schedule_write(path: Path, content: bytes) -> int:
        if path.name == "mission_schedule.csv":
            raise OSError("injected write failure")
        return original_write_bytes(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_during_schedule_write)
    with pytest.raises(OSError, match="injected write failure"):
        write_mission_scheduling_artifacts(result, destination)
    assert not destination.exists()
    assert list(parent.iterdir()) == []
