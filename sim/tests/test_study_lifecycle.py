from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import sim.analysis.study_lifecycle as study_lifecycle
from sim.analysis.study_lifecycle import (
    CAPABILITY_CONTRACTS,
    STUDY_CLAIMS_SCHEMA,
    STUDY_PLAN_SCHEMA,
    STUDY_REQUEST_SCHEMA,
    StudyClaims,
    StudyLifecycleError,
    StudyPlan,
    StudyRequest,
    build_study_bundle,
    compare_study_bundles,
    inspect_study_bundle,
    replay_study_bundle,
    verify_study_bundle,
)
from sim.installation.cli import _dispatch_commands
from sim.study import main as study_main


def _request(capabilities: tuple[str, ...] = ("trajectory_targeting",)) -> dict:
    return {
        "schema_version": STUDY_REQUEST_SCHEMA,
        "study_id": "portable-study-v1",
        "title": "Portable bounded study",
        "question": "Do the declared analyses satisfy their bounded acceptance criteria?",
        "capabilities": list(reversed(capabilities)),
        "assumptions": ["Inputs are synthetic and public-safe.", "Domain evidence is already complete."],
        "clarifications": [
            {"question": "Does study replay rerun physics?", "resolution": "No; domain replay remains separate."}
        ],
        "context": {
            "epoch": "relative elapsed seconds",
            "time_system": "UTC when absolute; otherwise elapsed SI seconds",
            "frame": "EME2000/ECI as declared by each domain record",
            "units": "OEL domain-contract units",
        },
        "fidelity": {
            "level": "bounded_public",
            "description": "Deterministic public analysis inside each named capability envelope.",
        },
        "acceptance_criteria": [
            {
                "criterion_id": f"criterion-{capability.replace('_', '-')}",
                "description": f"The {capability} evidence has its accepted terminal status.",
            }
            for capability in capabilities
        ],
    }


def _plan(capabilities: tuple[str, ...] = ("trajectory_targeting",)) -> dict:
    return {
        "schema_version": STUDY_PLAN_SCHEMA,
        "study_id": "portable-study-v1",
        "request_sha256": "auto",
        "resource_profile": "laptop-safe",
        "steps": [
            {
                "step_id": capability.replace("_", "-"),
                "capability": capability,
                "analysis_interface": CAPABILITY_CONTRACTS[capability]["analysis_interface"],
                "expected_evidence_schema": CAPABILITY_CONTRACTS[capability]["evidence_schema"],
                "depends_on": [],
                "acceptance_criterion_ids": [f"criterion-{capability.replace('_', '-')}"],
            }
            for capability in reversed(capabilities)
        ],
    }


def _claims(capabilities: tuple[str, ...] = ("trajectory_targeting",)) -> dict:
    return {
        "schema_version": STUDY_CLAIMS_SCHEMA,
        "study_id": "portable-study-v1",
        "plan_sha256": "auto",
        "claims": [
            {
                "claim_id": f"claim-{capability.replace('_', '-')}",
                "statement": f"The retained {capability} workflow reached its accepted bounded status.",
                "validation_level": "VC-1",
                "criterion_ids": [f"criterion-{capability.replace('_', '-')}"],
                "evidence": [{"step_id": capability.replace("_", "-"), "json_pointer": "/status"}],
            }
            for capability in reversed(capabilities)
        ],
        "non_claims": [
            "The study receipt does not authorize operational execution.",
            "Study replay does not replace authoritative domain-physics replay.",
        ],
    }


def _evidence(capability: str, *, marker: str = "baseline") -> dict:
    contract = CAPABILITY_CONTRACTS[capability]
    common = {
        "schema_version": contract["evidence_schema"],
        "status": contract["accepted_statuses"][0],
        "marker": marker,
    }
    digest = "a" * 64
    if capability == "constellation_design":
        return {
            **common,
            "analysis_id": "bounded-constellation-trade",
            "input_semantic_sha256": digest,
            "result_semantic_sha256": digest,
            "ranking": ["design-a"],
            "recommended_design_id": "design-a",
            "candidate_results": [
                {
                    "design_id": "design-a",
                    "rank": 1,
                    "feasible": True,
                    "score": 0.5,
                    "score_components": {"coverage_service": 0.5},
                    "generated_members": [{"member_id": "sat-1"}],
                    "coverage": {
                        "time_weighted_mean_covered_fraction": 0.5,
                        "interval_semantic_sha256": digest,
                    },
                    "network": {"union_sampled_available_fraction": 0.25},
                }
            ],
            "resource_estimate": {
                "candidate_count": 1,
                "sample_count": 2,
                "total_satellite_candidates": 2,
                "coverage_cell_time_comparisons": 100,
                "link_samples": 4,
            },
            "claim_limits": ["Synthetic structural test evidence only."],
        }
    if capability == "trajectory_targeting":
        return {
            **common,
            "problem_name": "bounded-target",
            "problem_sha256": digest,
            "converged": True,
            "variables": [{"name": "delta_v_i_m_s"}],
            "constraints": [{"name": "terminal-radius"}],
            "decision_values": [1.0],
            "solution_execution": {"final_state_eci_km_km_s": [1.0] * 6},
            "solution_constraint_evaluation": {"all_satisfied": True},
            "authoritative_repropagation": {
                "status": "verified",
                "execution": {"final_state_eci_km_km_s": [1.0] * 6},
                "constraint_evaluation": {"all_satisfied": True},
            },
            "resources": {"trajectory_evaluations": 1},
            "limitations": ["Synthetic structural test evidence only."],
        }
    if capability == "conjunction_assessment":
        return {
            **common,
            "problem_name": "bounded-conjunction",
            "problem_sha256": digest,
            "baseline": {
                "primary_id": "primary",
                "secondary_id": "secondary",
                "closest_approach": {
                    "time_s": 1.0,
                    "miss_distance_km": 0.1,
                    "relative_speed_km_s": 1.0,
                },
                "encounter_frame": {},
                "covariance_projection": {},
                "probability": {"collision_probability": 0.01},
            },
            "avoidance_candidates": [],
            "resources": {
                "primary_samples": 2,
                "secondary_samples": 2,
                "screening_object_count": 0,
                "candidate_count": 0,
            },
            "limitations": ["Synthetic structural test evidence only."],
        }
    if capability == "mission_scheduling":
        return {
            **common,
            "analysis_id": "bounded-schedule",
            "solver": "deterministic_exact_exhaustive_enumeration",
            "candidate_count": 1,
            "asset_count": 1,
            "station_count": 0,
            "evaluated_subset_count": 2,
            "feasible_subset_count": 2,
            "selected_count": 1,
            "selected_observation_count": 1,
            "objective_value": 1.0,
            "input_semantic_sha256": digest,
            "schedule_semantic_sha256": digest,
            "source_product_sha256s": [],
            "claim_limits": ["Synthetic structural test evidence only."],
        }
    if capability == "orbit_lifetime":
        return {
            **common,
            "analysis_id": "bounded-lifetime",
            "asset_id": "satellite",
            "outcome": "horizon_reached",
            "problem_semantic_sha256": digest,
            "result_semantic_sha256": digest,
            "propagator": {"family": "ONP"},
            "resource_use": {
                "integration_steps": 1,
                "output_samples": 2,
                "event_count": 0,
                "propagated_duration_s": 1.0,
            },
            "initial": {"altitude_km": 400.0},
            "final": {"time_s": 1.0},
            "thresholds": {},
            "claim_limits": ["Synthetic structural test evidence only."],
        }
    if capability == "spacecraft_power":
        return {
            **common,
            "analysis_id": "bounded-power",
            "asset_id": "satellite",
            "feasibility": "feasible",
            "model": "deterministic_sampled_solar_array_battery_v1",
            "problem_semantic_sha256": digest,
            "history_semantic_sha256": digest,
            "result_semantic_sha256": digest,
            "sample_count": 2,
            "illumination_interval_count": 1,
            "event_count": 0,
            "totals": {},
            "battery": {
                "initial_soc_fraction": 0.5,
                "final_soc_fraction": 0.5,
                "minimum_soc_fraction": 0.5,
                "maximum_soc_fraction": 0.5,
            },
            "conservation_residuals_wh": {
                "battery_storage": 0.0,
                "power_bus": 0.0,
                "load_service": 0.0,
            },
            "source_product_sha256s": [],
            "claim_limits": ["Synthetic structural test evidence only."],
        }
    raise AssertionError(f"Unhandled test capability: {capability}")


def _write_sources(root: Path, capabilities: tuple[str, ...], *, marker: str = "baseline") -> dict[str, Path]:
    root.mkdir(parents=True)
    sources = {}
    for capability in capabilities:
        step_id = capability.replace("_", "-")
        path = root / f"{step_id}.json"
        path.write_text(json.dumps(_evidence(capability, marker=marker), indent=2) + "\n", encoding="utf-8")
        sources[step_id] = path
    return sources


def _build(root: Path, capabilities: tuple[str, ...] = ("trajectory_targeting",), *, marker: str = "baseline"):
    sources = _write_sources(root / "sources", capabilities, marker=marker)
    return build_study_bundle(
        _request(capabilities),
        _plan(capabilities),
        _claims(capabilities),
        sources,
        root / "bundle",
    )


def _lifecycle_schema() -> dict:
    path = Path(__file__).resolve().parents[2] / "docs/contracts/schemas/oel-study-lifecycle-v1.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_request_plan_and_claims_normalize_and_bind_deterministically() -> None:
    capabilities = (
        "trajectory_targeting",
        "conjunction_assessment",
        "mission_scheduling",
        "orbit_lifetime",
        "spacecraft_power",
    )
    request = StudyRequest.from_mapping(_request(capabilities))
    plan = StudyPlan.from_mapping(_plan(capabilities), request)

    assert request.to_dict()["capabilities"] == sorted(capabilities)
    assert plan.to_dict()["request_sha256"] != "auto"
    assert [item["step_id"] for item in plan.to_dict()["steps"]] == sorted(
        capability.replace("_", "-") for capability in capabilities
    )

    invalid = _request()
    invalid["title"] = 123
    with pytest.raises(StudyLifecycleError, match="title must be a string"):
        StudyRequest.from_mapping(invalid)


def test_typed_records_are_revalidated_against_the_build_request(tmp_path: Path) -> None:
    first_request = StudyRequest.from_mapping(_request())
    first_plan = StudyPlan.from_mapping(_plan(), first_request)
    first_claims = StudyClaims.from_mapping(_claims(), first_request, first_plan)
    changed_request_value = _request()
    changed_request_value["question"] = "A different normalized question."
    changed_request = StudyRequest.from_mapping(changed_request_value)
    sources = _write_sources(tmp_path / "sources", ("trajectory_targeting",))

    with pytest.raises(StudyLifecycleError, match="request_sha256"):
        build_study_bundle(
            changed_request,
            first_plan,
            first_claims,
            sources,
            tmp_path / "mismatched",
        )


def test_mutated_typed_request_cannot_produce_an_unreplayable_bundle(tmp_path: Path) -> None:
    request = StudyRequest.from_mapping(_request())
    plan = StudyPlan.from_mapping(_plan(), request)
    claims = StudyClaims.from_mapping(_claims(), request, plan)
    request.payload["title"] = 123
    sources = _write_sources(tmp_path / "sources", ("trajectory_targeting",))

    with pytest.raises(StudyLifecycleError, match="title must be a string"):
        build_study_bundle(request, plan, claims, sources, tmp_path / "bundle")


def test_five_capability_bundle_build_inspect_and_identity_replay(tmp_path: Path) -> None:
    capabilities = (
        "trajectory_targeting",
        "conjunction_assessment",
        "mission_scheduling",
        "orbit_lifetime",
        "spacecraft_power",
    )
    artifacts = _build(tmp_path, capabilities)

    verification = verify_study_bundle(artifacts.output_dir)
    inspection = inspect_study_bundle(artifacts.output_dir)
    replay = replay_study_bundle(artifacts.output_dir)

    assert verification["status"] == "verified"
    assert verification["step_count"] == 5
    assert verification["claim_count"] == 5
    assert inspection["capabilities"] == sorted(capabilities)
    assert replay["replay_status"] == "identity_verified"
    assert {path.name for path in (artifacts.output_dir / "evidence").iterdir()} == {
        "conjunction-assessment.json",
        "mission-scheduling.json",
        "orbit-lifetime.json",
        "spacecraft-power.json",
        "trajectory-targeting.json",
    }


def test_published_schema_accepts_every_generated_lifecycle_record(tmp_path: Path) -> None:
    schema = _lifecycle_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    artifacts = _build(
        tmp_path,
        (
            "trajectory_targeting",
            "conjunction_assessment",
            "mission_scheduling",
            "orbit_lifetime",
            "spacecraft_power",
        ),
    )

    for path in (
        artifacts.request_json,
        artifacts.plan_json,
        artifacts.run_json,
        artifacts.evidence_json,
        artifacts.claims_json,
        artifacts.receipt_json,
    ):
        validator.validate(json.loads(path.read_text(encoding="utf-8")))


def test_bundle_rejects_tampered_evidence_and_root_records(tmp_path: Path) -> None:
    artifacts = _build(tmp_path)
    retained = artifacts.output_dir / "evidence" / "trajectory-targeting.json"
    retained.write_text(json.dumps(_evidence("trajectory_targeting", marker="tampered")), encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="Study evidence record"):
        verify_study_bundle(artifacts.output_dir)

    other = _build(tmp_path / "other")
    request = json.loads(other.request_json.read_text(encoding="utf-8"))
    request["question"] = "Changed after receipt creation."
    other.request_json.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="request_sha256|receipt"):
        verify_study_bundle(other.output_dir)


def test_bundle_rejects_unexpected_root_and_evidence_artifacts(tmp_path: Path) -> None:
    artifacts = _build(tmp_path / "root-extra")
    (artifacts.output_dir / "notes.txt").write_text("not part of the contract\n", encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="unexpected root artifact set"):
        verify_study_bundle(artifacts.output_dir)

    other = _build(tmp_path / "evidence-extra")
    (other.output_dir / "evidence" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="unexpected artifact set"):
        verify_study_bundle(other.output_dir)


def test_build_fails_closed_on_status_schema_binding_and_citation_errors(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path / "sources", ("trajectory_targeting",))
    source = sources["trajectory-targeting"]
    invalid = _evidence("trajectory_targeting")
    invalid["status"] = "non_convergent"
    source.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="non-accepted status"):
        build_study_bundle(_request(), _plan(), _claims(), sources, tmp_path / "bad-status")

    invalid["status"] = "converged"
    invalid["schema_version"] = "oel.unrelated.v1"
    source.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(StudyLifecycleError, match="has schema"):
        build_study_bundle(_request(), _plan(), _claims(), sources, tmp_path / "bad-schema")

    source.write_text(json.dumps(_evidence("trajectory_targeting")), encoding="utf-8")
    claims = _claims()
    claims["claims"][0]["evidence"][0]["json_pointer"] = "/missing"
    with pytest.raises(StudyLifecycleError, match="missing object member"):
        build_study_bundle(_request(), _plan(), claims, sources, tmp_path / "bad-pointer")


@pytest.mark.parametrize("capability", tuple(sorted(CAPABILITY_CONTRACTS)))
def test_build_rejects_schema_status_only_domain_evidence(tmp_path: Path, capability: str) -> None:
    capabilities = (capability,)
    step_id = capability.replace("_", "-")
    sources = _write_sources(tmp_path / "sources", capabilities)
    sources[step_id].write_text(
        json.dumps(
            {
                "schema_version": CAPABILITY_CONTRACTS[capability]["evidence_schema"],
                "status": CAPABILITY_CONTRACTS[capability]["accepted_statuses"][0],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StudyLifecycleError, match="missing required fields"):
        build_study_bundle(
            _request(capabilities),
            _plan(capabilities),
            _claims(capabilities),
            sources,
            tmp_path / "bundle",
        )


def test_claim_criteria_must_be_covered_by_cited_steps_and_validation_level(tmp_path: Path) -> None:
    capabilities = ("trajectory_targeting", "conjunction_assessment")
    sources = _write_sources(tmp_path / "sources", capabilities)
    mismatched = _claims(capabilities)
    conjunction_claim = next(
        item for item in mismatched["claims"] if item["claim_id"] == "claim-conjunction-assessment"
    )
    conjunction_claim["criterion_ids"] = ["criterion-trajectory-targeting"]

    with pytest.raises(StudyLifecycleError, match="covered by a cited evidence step"):
        build_study_bundle(
            _request(capabilities),
            _plan(capabilities),
            mismatched,
            sources,
            tmp_path / "mismatched",
        )

    overclaimed = _claims()
    overclaimed["claims"][0]["validation_level"] = "VC-4"
    with pytest.raises(StudyLifecycleError, match="exceeds the cited evidence-step maximum VC-1"):
        build_study_bundle(
            _request(),
            _plan(),
            overclaimed,
            {"trajectory-targeting": sources["trajectory-targeting"]},
            tmp_path / "overclaimed",
        )


def test_plan_rejects_cycles_unknown_fields_and_uncovered_criteria() -> None:
    capabilities = ("trajectory_targeting", "conjunction_assessment")
    request = _request(capabilities)
    cyclic = _plan(capabilities)
    cyclic["steps"][0]["depends_on"] = [cyclic["steps"][1]["step_id"]]
    cyclic["steps"][1]["depends_on"] = [cyclic["steps"][0]["step_id"]]
    with pytest.raises(StudyLifecycleError, match="cycle"):
        StudyPlan.from_mapping(cyclic, request)

    unknown = _request()
    unknown["surprise"] = True
    with pytest.raises(StudyLifecycleError, match="unknown fields"):
        StudyRequest.from_mapping(unknown)

    uncovered = _plan()
    uncovered["steps"][0]["acceptance_criterion_ids"] = []
    with pytest.raises(StudyLifecycleError, match="must not be empty|does not cover"):
        StudyPlan.from_mapping(uncovered, _request())


def test_schedule_to_power_dependency_requires_matching_source_digest(tmp_path: Path) -> None:
    capabilities = ("mission_scheduling", "spacecraft_power")
    request = _request(capabilities)
    plan = _plan(capabilities)
    power_step = next(step for step in plan["steps"] if step["capability"] == "spacecraft_power")
    power_step["depends_on"] = ["mission-scheduling"]
    sources = _write_sources(tmp_path / "sources", capabilities)

    with pytest.raises(StudyLifecycleError, match="does not bind the retained schedule"):
        build_study_bundle(request, plan, _claims(capabilities), sources, tmp_path / "unbound")

    power = json.loads(sources["spacecraft-power"].read_text())
    power["source_product_sha256s"] = ["a" * 64]
    sources["spacecraft-power"].write_text(json.dumps(power) + "\n")
    bundle = build_study_bundle(request, plan, _claims(capabilities), sources, tmp_path / "bound")
    assert inspect_study_bundle(bundle.output_dir)["status"] == "verified"


def test_comparison_reports_equivalent_and_changed_evidence(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")
    changed = _build(tmp_path / "changed", marker="changed")

    equivalent = compare_study_bundles(first.output_dir, second.output_dir)
    difference = compare_study_bundles(first.output_dir, changed.output_dir)

    assert equivalent["status"] == "equivalent"
    assert equivalent["same_bundle"] is True
    assert difference["status"] == "different"
    assert difference["changed_evidence_steps"] == ["trajectory-targeting"]
    assert "study_evidence.json" in difference["changed_records"]


def test_cli_build_inspect_replay_compare_and_error_paths(tmp_path: Path, capsys) -> None:
    request_path = tmp_path / "request.json"
    plan_path = tmp_path / "plan.json"
    claims_path = tmp_path / "claims.json"
    source_path = tmp_path / "source.json"
    for path, value in (
        (request_path, _request()),
        (plan_path, _plan()),
        (claims_path, _claims()),
        (source_path, _evidence("trajectory_targeting")),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    bundle = tmp_path / "bundle"

    assert (
        study_main(
            [
                "build",
                str(request_path),
                str(plan_path),
                str(claims_path),
                "--evidence",
                f"trajectory-targeting={source_path}",
                "--output-dir",
                str(bundle),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert study_main(["inspect", str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert study_main(["replay", str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out)["replay_status"] == "identity_verified"
    assert study_main(["compare", str(bundle), str(bundle)]) == 0
    assert json.loads(capsys.readouterr().out)["same_bundle"] is True
    assert (
        study_main(
            [
                "build",
                str(request_path),
                str(plan_path),
                str(claims_path),
                "--evidence",
                "bad",
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_output_directory_and_evidence_bindings_fail_closed(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path / "sources", ("trajectory_targeting",))
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(StudyLifecycleError, match="must not already exist"):
        build_study_bundle(_request(), _plan(), _claims(), sources, occupied)
    with pytest.raises(StudyLifecycleError, match="exactly match"):
        build_study_bundle(_request(), _plan(), _claims(), {}, tmp_path / "missing")


def test_build_and_verify_reject_explicit_symbolic_links(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path / "sources", ("trajectory_targeting",))
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(sources["trajectory-targeting"])
    with pytest.raises(StudyLifecycleError, match="must not be a symbolic link"):
        build_study_bundle(
            _request(),
            _plan(),
            _claims(),
            {"trajectory-targeting": source_link},
            tmp_path / "source-link-bundle",
        )

    artifacts = _build(tmp_path / "real")
    bundle_link = tmp_path / "bundle-link"
    bundle_link.symlink_to(artifacts.output_dir, target_is_directory=True)
    with pytest.raises(StudyLifecycleError, match="must not be a symbolic link"):
        verify_study_bundle(bundle_link)


def test_verify_rejects_evidence_directory_and_file_symlinks_before_reading(tmp_path: Path) -> None:
    directory_bundle = _build(tmp_path / "directory").output_dir
    evidence_root = directory_bundle / "evidence"
    external_evidence = tmp_path / "external-evidence"
    evidence_root.rename(external_evidence)
    evidence_root.symlink_to(external_evidence, target_is_directory=True)
    with pytest.raises(StudyLifecycleError, match="Study evidence path must be a regular directory"):
        verify_study_bundle(directory_bundle)

    file_bundle = _build(tmp_path / "file").output_dir
    evidence_file = file_bundle / "evidence" / "trajectory-targeting.json"
    external_file = tmp_path / "external-evidence.json"
    evidence_file.rename(external_file)
    evidence_file.symlink_to(external_file)
    with pytest.raises(StudyLifecycleError, match="Study evidence artifact must be a regular file"):
        verify_study_bundle(file_bundle)


def test_verify_rejects_evidence_replaced_during_verification(monkeypatch, tmp_path: Path) -> None:
    artifacts = _build(tmp_path / "race")
    retained = artifacts.output_dir / "evidence" / "trajectory-targeting.json"
    original = study_lifecycle._evidence_record
    calls = 0

    def mutate_after_first_read(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            value = json.loads(retained.read_text(encoding="utf-8"))
            value["problem_name"] = "tampered-after-evidence-read"
            retained.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(study_lifecycle, "_evidence_record", mutate_after_first_read)
    with pytest.raises(StudyLifecycleError, match="evidence changed during verification"):
        verify_study_bundle(artifacts.output_dir)
    assert calls == 2


def test_study_is_a_unified_oel_dispatch_command() -> None:
    assert "study" in _dispatch_commands()


def test_claim_input_order_does_not_change_bundle_identity(tmp_path: Path) -> None:
    capabilities = ("trajectory_targeting", "conjunction_assessment")
    request = _request(capabilities)
    plan = _plan(capabilities)
    claims = _claims(capabilities)
    sources = _write_sources(tmp_path / "sources", capabilities)
    first = build_study_bundle(request, plan, claims, sources, tmp_path / "first")

    reordered_request = deepcopy(request)
    reordered_request["capabilities"].reverse()
    reordered_request["assumptions"].reverse()
    reordered_plan = deepcopy(plan)
    reordered_plan["steps"].reverse()
    reordered_claims = deepcopy(claims)
    reordered_claims["claims"].reverse()
    reordered_claims["non_claims"].reverse()
    second = build_study_bundle(reordered_request, reordered_plan, reordered_claims, sources, tmp_path / "second")

    assert compare_study_bundles(first.output_dir, second.output_dir)["same_bundle"] is True
