from __future__ import annotations

from copy import deepcopy

from sim.study_planning.preflight import PlanningResourcePolicy, _resource_findings
from sim.study_planning.resource_accounting import ResourceAccounting, account_plan_resources
from sim.study_planning.schema_validation import validate_schema
from sim.study_planning.schemas import RESOURCE_ACCOUNTING_SCHEMA


def _operation(
    operation_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    cases: int = 1,
    wall_time_s: float = 1.0,
    cpu_time_s: float | None = None,
    work_units: float | None = None,
    propagation_steps: int | None = None,
    estimator_evaluations: int | None = None,
    batch_evaluations: int | None = None,
    arc_duration_s: float | None = None,
    memory_mb: float | None = None,
    retained_output_bytes: int | None = None,
    input_refs: tuple[dict, ...] = (),
    output_refs: tuple[dict, ...] = (),
) -> dict:
    bounds = {"max_cases": cases, "max_wall_time_s": wall_time_s}
    if cpu_time_s is not None:
        bounds["max_cpu_time_s"] = cpu_time_s
    if work_units is not None:
        bounds["max_work_units"] = work_units
    if propagation_steps is not None:
        bounds["max_propagation_steps"] = propagation_steps
    if estimator_evaluations is not None:
        bounds["max_estimator_evaluations"] = estimator_evaluations
    if batch_evaluations is not None:
        bounds["max_batch_evaluations"] = batch_evaluations
    if arc_duration_s is not None:
        bounds["max_arc_duration_s"] = arc_duration_s
    if memory_mb is not None:
        bounds["max_peak_memory_mb"] = memory_mb
    if retained_output_bytes is not None:
        bounds["max_retained_output_bytes"] = retained_output_bytes
    return {
        "operation_id": operation_id,
        "depends_on": list(depends_on),
        "bounds": bounds,
        "input_refs": list(input_refs),
        "output_refs": list(output_refs),
    }


def _input(ref_id: str, digest: str | None, byte_count: object) -> dict:
    metadata = {"input_bytes": byte_count}
    if digest is not None:
        metadata["content_sha256"] = digest
    return {"ref_id": ref_id, "metadata": metadata}


def _counted_input(
    ref_id: str,
    digest: str | None,
    *,
    observation_count: object | None = None,
    normal_point_count: object | None = None,
    sample_count: object | None = None,
    station_count: object | None = None,
) -> dict:
    metadata = {}
    if digest is not None:
        metadata["content_sha256"] = digest
    for key, value in (
        ("observation_count", observation_count),
        ("normal_point_count", normal_point_count),
        ("sample_count", sample_count),
        ("station_count", station_count),
    ):
        if value is not None:
            metadata[key] = value
    return {"ref_id": ref_id, "metadata": metadata}


def _output(ref_id: str, retained_bytes: object | None) -> dict:
    metadata = {}
    if retained_bytes is not None:
        metadata["retained_bytes"] = retained_bytes
    return {"ref_id": ref_id, "metadata": metadata}


def _derived_output(ref_id: str, producer_id: str, semantic_role: str) -> dict:
    return {
        "ref_id": ref_id,
        "kind": "derived_artifact",
        "schema_id": "oel.artifact.derived.v1",
        "metadata": {
            "producer_operation_id": producer_id,
            "semantic_role": semantic_role,
        },
    }


def test_resource_accounting_preserves_legacy_positional_constructor() -> None:
    accounting = ResourceAccounting(
        1,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        1,
        (),
    )

    assert accounting.worker_bound == 1
    assert accounting.issues == ()
    assert accounting.observation_count is None
    assert accounting.station_count is None


def test_resource_accounting_schema_reads_legacy_packets_without_new_metrics() -> None:
    packet = account_plan_resources(
        [_operation("legacy")],
        {"max_parallel_workers": 1},
    ).to_dict()
    packet["metrics"].pop("observation_count")
    packet["metrics"].pop("station_count")

    assert validate_schema(packet, RESOURCE_ACCOUNTING_SCHEMA) == ()


def test_accounting_sums_cpu_work_cases_and_unique_content_outputs() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    operations = [
        _operation(
            "a",
            cases=2,
            cpu_time_s=3.0,
            work_units=5.0,
            memory_mb=4.0,
            retained_output_bytes=40,
            input_refs=(_input("input-a", digest_a, 100),),
            output_refs=(_output("output-a", 40),),
        ),
        _operation(
            "b",
            cases=3,
            cpu_time_s=7.0,
            work_units=8.0,
            memory_mb=6.0,
            retained_output_bytes=50,
            input_refs=(
                _input("input-a-again", digest_a, 100),
                _input("input-b", digest_b, 50),
            ),
            output_refs=(_output("output-b", 50),),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 2})

    assert accounting.total_cases == 5
    assert accounting.cpu_time_s == 10.0
    assert accounting.work_units == 13.0
    assert accounting.peak_memory_mb == 10.0
    assert accounting.unique_input_bytes == 150
    assert accounting.retained_output_bytes == 90
    assert accounting.issues == ()
    summary = accounting.to_dict()
    assert summary["metrics"]["cpu_time_s"]["status"] == "assessed"
    assert summary["metrics"]["unique_input_bytes"]["status"] == "assessed"
    assert summary["metrics"]["retained_output_bytes"]["status"] == "assessed"


def test_typed_work_bounds_are_separate_cumulative_metrics() -> None:
    operations = [
        _operation(
            "first",
            propagation_steps=60,
            estimator_evaluations=7,
            batch_evaluations=4,
        ),
        _operation(
            "second",
            propagation_steps=60,
            estimator_evaluations=7,
            batch_evaluations=4,
        ),
    ]

    accounting = account_plan_resources(
        operations,
        {
            "max_parallel_workers": 1,
            "max_propagation_steps": 100,
            "max_estimator_evaluations": 20,
            "max_batch_evaluations": 20,
        },
    )

    assert accounting.propagation_steps == 120
    assert accounting.estimator_evaluations == 14
    assert accounting.batch_evaluations == 8
    assert accounting.to_dict()["metrics"]["propagation_steps"]["status"] == "assessed"
    assert accounting.to_dict()["metrics"]["estimator_evaluations"]["status"] == "assessed"
    assert accounting.to_dict()["metrics"]["batch_evaluations"]["status"] == "assessed"

    plan = {
        "resource_envelope": {
            "max_operations": 2,
            "max_total_cases": 2,
            "max_wall_time_s": 10.0,
            "max_cpu_time_s": 10.0,
            "max_peak_memory_mb": 10.0,
            "max_storage_mb": 1.0,
            "max_artifact_mb": 1.0,
            "max_parallel_workers": 1,
            "max_propagation_steps": 100,
            "max_estimator_evaluations": 20,
            "max_batch_evaluations": 20,
        }
    }
    codes = {
        finding.code
        for finding in _resource_findings(
            plan,
            operations,
            PlanningResourcePolicy(
                max_operations=4,
                max_total_cases=10,
                max_wall_time_s=10.0,
                max_cpu_time_s=10.0,
                max_peak_memory_mb=10.0,
                max_storage_mb=10.0,
                max_artifact_mb=10.0,
                max_parallel_workers=4,
                max_work_units=10.0,
                max_propagation_steps=200,
                max_estimator_evaluations=20,
                max_batch_evaluations=20,
                max_retained_output_bytes=10_000,
            ),
        )
    }
    assert "resource.propagation_steps_accounted_exceeded" in codes
    assert "resource.estimator_evaluations_accounted_exceeded" not in codes
    assert "resource.batch_evaluations_accounted_exceeded" not in codes


def test_arc_duration_is_a_per_operation_maximum() -> None:
    accounting = account_plan_resources(
        [_operation("first", arc_duration_s=90.0), _operation("second", arc_duration_s=80.0)],
        {"max_parallel_workers": 1, "max_arc_duration_s": 100.0},
    )

    assert accounting.arc_duration_s == 90.0
    assert accounting.to_dict()["metrics"]["arc_duration_s"]["basis"].startswith("maximum")


def test_peak_memory_respects_dependencies_and_worker_bound() -> None:
    operations = [
        _operation("a", wall_time_s=5.0, memory_mb=4.0),
        _operation("b", wall_time_s=10.0, memory_mb=6.0),
        _operation("c", depends_on=("a",), wall_time_s=5.0, memory_mb=10.0),
    ]

    parallel = account_plan_resources(operations, {"max_parallel_workers": 2})
    serial = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert parallel.peak_memory_mb == 16.0
    assert serial.peak_memory_mb == 10.0
    assert parallel.issues == ()
    assert serial.issues == ()


def test_peak_memory_does_not_assume_max_duration_overlap() -> None:
    operations = [
        _operation("long-memory", wall_time_s=10.0, memory_mb=100.0),
        _operation("short-prerequisite", wall_time_s=100.0, memory_mb=1.0),
        _operation(
            "dependent-memory",
            depends_on=("short-prerequisite",),
            wall_time_s=10.0,
            memory_mb=100.0,
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 2})

    # The prerequisite can finish early while long-memory is still active.
    # A max-duration schedule would incorrectly report 101 MB here; the
    # dependency-incomparability bound preserves the legal 200 MB overlap.
    assert accounting.peak_memory_mb == 200.0
    assert accounting.issues == ()


def test_partial_metric_declarations_fail_closed() -> None:
    operations = [
        _operation(
            "declared",
            cpu_time_s=2.0,
            input_refs=(_input("bound-input", None, 10),),
            output_refs=(_output("retained-output", 20),),
            retained_output_bytes=20,
        ),
        _operation(
            "missing",
            input_refs=(_input("unbound-input", "c" * 64, 10),),
            output_refs=(_output("unknown-output", None),),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})
    codes = {issue.code for issue in accounting.issues}

    assert accounting.cpu_time_s is None
    assert accounting.unique_input_bytes is None
    assert accounting.retained_output_bytes is None
    assert "resource.cpu_time_s_estimate_missing" in codes
    assert "resource.input_content_identity_missing" in codes
    assert "resource.retained_output_estimate_missing" in codes


def test_unique_input_bytes_counts_external_content_once_across_derived_chain() -> None:
    producer_output = _derived_output("producer.derived", "producer", "derived_result")
    middle_output = _derived_output("middle.derived", "middle", "derived_result")
    operations = [
        _operation(
            "producer",
            input_refs=(_input("request", "a" * 64, 100),),
            output_refs=(producer_output,),
        ),
        _operation(
            "middle",
            depends_on=("producer",),
            input_refs=(deepcopy(producer_output),),
            output_refs=(middle_output,),
        ),
        _operation(
            "consumer",
            depends_on=("middle",),
            input_refs=(deepcopy(middle_output),),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.unique_input_bytes == 100
    assert accounting.issues == ()


def test_spoofed_derived_references_remain_external() -> None:
    producer_output = _derived_output("producer.derived", "producer", "derived_result")
    wrong_role = deepcopy(producer_output)
    wrong_role["metadata"]["semantic_role"] = "spoofed_result"
    operations = [
        _operation(
            "producer",
            input_refs=(_input("request", "b" * 64, 100),),
            output_refs=(producer_output,),
        ),
        _operation("missing-edge", input_refs=(deepcopy(producer_output),)),
        _operation(
            "wrong-metadata",
            depends_on=("producer",),
            input_refs=(wrong_role,),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.unique_input_bytes is None
    assert sum(issue.code == "resource.input_bytes_estimate_missing" for issue in accounting.issues) == 2


def test_malformed_self_ambiguous_or_duplicate_derived_refs_remain_external() -> None:
    producer_output = _derived_output("producer.derived", "producer", "derived_result")
    self_output = _derived_output("self.derived", "self", "self_result")
    shared_first = _derived_output("shared.derived", "first", "derived_result")
    shared_second = _derived_output("shared.derived", "second", "derived_result")
    operations = [
        _operation(
            "producer",
            input_refs=(_input("request", "d" * 64, 100),),
            output_refs=(producer_output,),
        ),
        _operation(
            "self",
            depends_on=("self",),
            input_refs=(deepcopy(self_output),),
            output_refs=(self_output,),
        ),
        _operation("first", output_refs=(shared_first,)),
        _operation("second", output_refs=(shared_second,)),
        _operation(
            "ambiguous-consumer",
            depends_on=("first", "second"),
            input_refs=(deepcopy(shared_first),),
        ),
        _operation(
            "duplicate-consumer",
            depends_on=("producer",),
            input_refs=(deepcopy(producer_output),),
        ),
        _operation(
            "duplicate-consumer",
            depends_on=("producer",),
            input_refs=(deepcopy(producer_output),),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.unique_input_bytes is None
    assert sum(issue.code == "resource.input_bytes_estimate_missing" for issue in accounting.issues) == 4


def test_partial_external_input_metadata_stays_fail_closed_after_derived_match() -> None:
    producer_output = _derived_output("producer.derived", "producer", "derived_result")
    operations = [
        _operation(
            "producer",
            input_refs=(_input("request", "c" * 64, 100),),
            output_refs=(producer_output,),
        ),
        _operation(
            "consumer",
            depends_on=("producer",),
            input_refs=(deepcopy(producer_output),),
        ),
        _operation("external", input_refs=(_input("partial", None, 25),)),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.unique_input_bytes is None
    assert "resource.input_content_identity_missing" in {
        issue.code for issue in accounting.issues
    }


def test_explicit_metric_caps_require_their_metadata() -> None:
    accounting = account_plan_resources(
        [_operation("unestimated")],
        {
            "max_parallel_workers": 1,
            "max_work_units": 10.0,
            "max_input_bytes": 100,
            "max_retained_output_bytes": 100,
        },
    )
    codes = {issue.code for issue in accounting.issues}

    assert accounting.work_units is None
    assert accounting.unique_input_bytes is None
    assert accounting.retained_output_bytes is None
    assert "resource.work_units_estimate_missing" in codes
    assert "resource.input_bytes_estimate_missing" in codes
    assert "resource.retained_output_estimate_missing" in codes
    assert accounting.to_dict()["metrics"]["work_units"]["status"] == "unassessed"


def test_observation_and_station_counts_deduplicate_exact_content_identity() -> None:
    digest_a = "a" * 64
    operations = [
        _operation(
            "first",
            input_refs=(
                _counted_input(
                    "observations-a",
                    digest_a,
                    observation_count=5,
                    station_count=2,
                ),
            ),
        ),
        _operation(
            "same-content",
            input_refs=(
                _counted_input(
                    "observations-a-again",
                    digest_a,
                    normal_point_count=5,
                    station_count=2,
                ),
            ),
        ),
        _operation(
            "different-content",
            input_refs=(
                _counted_input(
                    "observations-b",
                    "b" * 64,
                    sample_count=3,
                    station_count=1,
                ),
            ),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.observation_count == 8
    assert accounting.station_count == 3
    assert accounting.issues == ()
    assert accounting.to_dict()["metrics"]["observation_count"]["status"] == "assessed"
    assert accounting.to_dict()["metrics"]["station_count"]["status"] == "assessed"


def test_unresolved_observation_and_station_counts_remain_per_reference() -> None:
    operations = [
        _operation(
            "first",
            input_refs=(
                _counted_input(
                    "unresolved-a",
                    None,
                    observation_count=5,
                    station_count=2,
                ),
            ),
        ),
        _operation(
            "second",
            input_refs=(
                _counted_input(
                    "unresolved-b",
                    None,
                    observation_count=3,
                    station_count=2,
                ),
            ),
        ),
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})

    assert accounting.observation_count == 8
    assert accounting.station_count == 4
    assert accounting.issues == ()


def test_partial_or_conflicting_input_counts_fail_closed() -> None:
    digest = "c" * 64
    partial = account_plan_resources(
        [
            _operation(
                "declared",
                input_refs=(_counted_input("declared-ref", digest, observation_count=5),),
            ),
            _operation("missing", input_refs=(_counted_input("missing-ref", "d" * 64),)),
        ],
        {"max_parallel_workers": 1},
    )
    assert partial.observation_count is None
    assert "resource.observation_count_estimate_missing" in {
        issue.code for issue in partial.issues
    }

    conflicting_identity = account_plan_resources(
        [
            _operation(
                "first",
                input_refs=(_counted_input("same-a", digest, observation_count=5),),
            ),
            _operation(
                "second",
                input_refs=(_counted_input("same-b", digest, observation_count=6),),
            ),
        ],
        {"max_parallel_workers": 1},
    )
    assert conflicting_identity.observation_count is None
    assert "resource.observation_count_estimate_conflict" in {
        issue.code for issue in conflicting_identity.issues
    }

    conflicting_aliases = account_plan_resources(
        [
            _operation(
                "aliases",
                input_refs=(
                    _counted_input(
                        "alias-ref",
                        digest,
                        observation_count=5,
                        sample_count=4,
                    ),
                ),
            )
        ],
        {"max_parallel_workers": 1},
    )
    assert conflicting_aliases.observation_count is None
    assert "resource.observation_count_estimate_conflict" in {
        issue.code for issue in conflicting_aliases.issues
    }


def test_explicit_count_caps_require_metadata_and_block_accounted_overrun() -> None:
    unassessed = account_plan_resources(
        [_operation("missing")],
        {
            "max_parallel_workers": 1,
            "max_observations": 5,
            "max_stations": 1,
        },
    )
    unassessed_codes = {issue.code for issue in unassessed.issues}
    assert unassessed.observation_count is None
    assert unassessed.station_count is None
    assert "resource.observation_count_estimate_missing" in unassessed_codes
    assert "resource.station_count_estimate_missing" in unassessed_codes

    operations = [
        _operation(
            "counted",
            input_refs=(
                _counted_input(
                    "counted-ref",
                    "e" * 64,
                    observation_count=6,
                    station_count=2,
                ),
            ),
        )
    ]
    plan = {
        "resource_envelope": {
            "max_operations": 1,
            "max_total_cases": 1,
            "max_wall_time_s": 1.0,
            "max_cpu_time_s": 1.0,
            "max_peak_memory_mb": 1.0,
            "max_storage_mb": 1.0,
            "max_artifact_mb": 1.0,
            "max_parallel_workers": 1,
            "max_observations": 5,
            "max_stations": 1,
        }
    }
    findings = _resource_findings(plan, operations, PlanningResourcePolicy())
    codes = {finding.code for finding in findings}
    assert "resource.observation_count_accounted_exceeded" in codes
    assert "resource.station_count_accounted_exceeded" in codes


def test_legacy_plans_without_count_caps_remain_unassessed() -> None:
    accounting = account_plan_resources(
        [_operation("legacy")],
        {"max_parallel_workers": 1},
    )

    assert accounting.observation_count is None
    assert accounting.station_count is None
    assert accounting.issues == ()
    summary = accounting.to_dict()
    assert summary["metrics"]["observation_count"]["status"] == "unassessed"
    assert summary["metrics"]["station_count"]["status"] == "unassessed"


def test_invalid_metric_numbers_fail_closed() -> None:
    operations = [
        _operation(
            "invalid",
            cpu_time_s=float("nan"),
            work_units=-1.0,
            memory_mb=float("inf"),
            input_refs=(_input("bad-input", "d" * 64, True),),
            output_refs=(_output("bad-output", 1.5),),
            retained_output_bytes=2,
        )
    ]

    accounting = account_plan_resources(operations, {"max_parallel_workers": 1})
    codes = {issue.code for issue in accounting.issues}

    assert accounting.cpu_time_s is None
    assert accounting.work_units is None
    assert accounting.peak_memory_mb is None
    assert accounting.unique_input_bytes is None
    assert accounting.retained_output_bytes is None
    assert "resource.cpu_time_s_estimate_invalid" in codes
    assert "resource.work_units_estimate_invalid" in codes
    assert "resource.peak_memory_mb_estimate_invalid" in codes
    assert "resource.input_bytes_estimate_invalid" in codes
    assert "resource.retained_output_estimate_invalid" in codes


def test_preflight_retains_per_operation_bounds_and_checks_accounted_caps() -> None:
    operations = [
        _operation(
            "a",
            cases=2,
            cpu_time_s=4.0,
            work_units=5.0,
            memory_mb=6.0,
            retained_output_bytes=40,
            output_refs=(_output("output-a", 40),),
        )
    ]
    plan = {
        "resource_envelope": {
            "max_operations": 1,
            "max_total_cases": 1,
            "max_wall_time_s": 1.0,
            "max_cpu_time_s": 3.0,
            "max_work_units": 3.0,
            "max_peak_memory_mb": 5.0,
            "max_storage_mb": 1.0,
            "max_artifact_mb": 1.0,
            "max_parallel_workers": 1,
            "max_retained_output_bytes": 30,
        }
    }

    findings = _resource_findings(
        plan,
        operations,
        PlanningResourcePolicy(
            max_operations=4,
            max_total_cases=10,
            max_wall_time_s=10.0,
            max_cpu_time_s=10.0,
            max_peak_memory_mb=10.0,
            max_storage_mb=10.0,
            max_artifact_mb=10.0,
            max_parallel_workers=4,
            max_work_units=10.0,
            max_retained_output_bytes=10_000,
        ),
    )
    codes = {finding.code for finding in findings}

    assert "resource.total_cases_exceeded" in codes
    assert "resource.cpu_time_s_accounted_exceeded" in codes
    assert "resource.work_units_accounted_exceeded" in codes
    assert "resource.peak_memory_mb_accounted_exceeded" in codes
    assert "resource.retained_output_bytes_accounted_exceeded" in codes

    declared_over_policy = deepcopy(plan)
    declared_over_policy["resource_envelope"]["max_cpu_time_s"] = 20.0
    declared_over_policy["resource_envelope"]["max_peak_memory_mb"] = 20.0
    small_operations = [_operation("small", cpu_time_s=1.0, memory_mb=1.0)]
    declared_codes = {
        finding.code
        for finding in _resource_findings(
            declared_over_policy,
            small_operations,
            PlanningResourcePolicy(
                max_operations=4,
                max_total_cases=10,
                max_wall_time_s=10.0,
                max_cpu_time_s=10.0,
                max_peak_memory_mb=10.0,
                max_storage_mb=10.0,
                max_artifact_mb=10.0,
                max_parallel_workers=4,
                max_work_units=10.0,
                max_retained_output_bytes=10_000,
            ),
        )
    }
    assert "resource.cpu_time_s_exceeded" in declared_codes
    assert "resource.peak_memory_mb_exceeded" in declared_codes
    assert "resource.cpu_time_s_accounted_exceeded" not in declared_codes
