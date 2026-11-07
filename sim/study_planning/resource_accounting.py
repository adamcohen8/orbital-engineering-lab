"""Metric-specific resource accounting for content-bound study plans.

The planner only accounts for values explicitly declared by a plan.  It never
turns wall time into CPU time, guesses output sizes, or treats an unbound
artifact as a byte-identifiable input.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INPUT_BYTES_KEY = "input_bytes"
_RETAINED_BYTES_KEYS = ("retained_bytes", "retained_output_bytes")
_OBSERVATION_COUNT_KEYS = ("observation_count", "normal_point_count", "sample_count")
_STATION_COUNT_KEYS = ("station_count",)


@dataclass(frozen=True, slots=True)
class ResourceAccountingIssue:
    """A fail-closed accounting problem found in a proposed plan."""

    code: str
    message: str
    paths: tuple[str, ...]
    observed: Mapping[str, Any]
    suggestion: str


@dataclass(frozen=True, slots=True)
class ResourceAccounting:
    """Derived PLAN resource totals.

    None means that the plan did not opt into that metric.  A metric is also
    None when an opt-in declaration is incomplete or invalid; the corresponding
    issue explains why the total is unavailable.
    """

    total_cases: int
    cpu_time_s: float | None
    work_units: float | None
    propagation_steps: int | None
    estimator_evaluations: int | None
    batch_evaluations: int | None
    arc_duration_s: float | None
    peak_memory_mb: float | None
    unique_input_bytes: int | None
    retained_output_bytes: int | None
    worker_bound: int
    issues: tuple[ResourceAccountingIssue, ...] = ()
    observation_count: int | None = None
    station_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return an inspectable assessed/unassessed accounting summary."""

        def metric(
            value: float | int | None,
            basis: str,
            prefixes: tuple[str, ...],
        ) -> dict[str, Any]:
            issue_codes = [
                issue.code
                for issue in self.issues
                if any(issue.code.startswith(prefix) for prefix in prefixes)
            ]
            return {
                "status": "assessed" if value is not None else "unassessed",
                "value": value,
                "basis": basis,
                "issue_codes": issue_codes,
            }

        return {
            "schema": "oel.study_resource_accounting.v1",
            "metrics": {
                "total_cases": metric(
                    self.total_cases,
                    "sum of operation bounds.max_cases",
                    ("resource.total_cases",),
                ),
                "cpu_time_s": metric(
                    self.cpu_time_s,
                    "sum of operation bounds.max_cpu_time_s",
                    ("resource.cpu_time_s",),
                ),
                "work_units": metric(
                    self.work_units,
                    "sum of operation bounds.max_work_units",
                    ("resource.work_units",),
                ),
                "propagation_steps": metric(
                    self.propagation_steps,
                    "sum of operation bounds.max_propagation_steps",
                    ("resource.propagation_steps",),
                ),
                "estimator_evaluations": metric(
                    self.estimator_evaluations,
                    "sum of operation bounds.max_estimator_evaluations",
                    ("resource.estimator_evaluations",),
                ),
                "batch_evaluations": metric(
                    self.batch_evaluations,
                    "sum of operation bounds.max_batch_evaluations",
                    ("resource.batch_evaluations",),
                ),
                "arc_duration_s": metric(
                    self.arc_duration_s,
                    "maximum of per-operation bounds.max_arc_duration_s",
                    ("resource.arc_duration_s",),
                ),
                "peak_memory_mb": metric(
                    self.peak_memory_mb,
                    "conservative dependency-incomparable bound of operation bounds.max_peak_memory_mb",
                    ("resource.peak_memory_mb", "resource.memory_"),
                ),
                "unique_input_bytes": metric(
                    self.unique_input_bytes,
                    "unique externally supplied metadata.content_sha256 values and metadata.input_bytes; authenticated derived outputs are excluded",
                    ("resource.input_",),
                ),
                "retained_output_bytes": metric(
                    self.retained_output_bytes,
                    "sum of output metadata.retained_bytes values",
                    ("resource.retained_output_",),
                ),
                "observation_count": metric(
                    self.observation_count,
                    "unique metadata.content_sha256 input identities; unresolved references counted per reference using observation_count, normal_point_count, or sample_count",
                    ("resource.observation_count",),
                ),
                "station_count": metric(
                    self.station_count,
                    "unique metadata.content_sha256 input identities; unresolved references counted per reference using station_count",
                    ("resource.station_count",),
                ),
            },
            "worker_bound": self.worker_bound,
            "issue_codes": [issue.code for issue in self.issues],
        }


def account_plan_resources(
    operations: Sequence[Mapping[str, Any]],
    envelope: Mapping[str, Any],
    *,
    countable_input_refs: Mapping[str, set[tuple[int, int]]] | None = None,
) -> ResourceAccounting:
    """Derive declared resource totals without inventing estimates.

    Operation CPU time, work, and peak memory are declared in bounds.
    Externally supplied input byte counts and content identities live in
    input-ref metadata.  An authenticated direct dependency output is excluded
    from this metric because its retained bytes are accounted separately.
    Retained output sizes live in output-ref metadata.  The output metric is
    intentionally opt-in because older plans do not carry output-size
    estimates.  Preflight may supply catalog-qualified input-reference sets for
    count metrics so typed problem/configuration ports without that metric in
    their authoritative required metadata are not mistaken for datasets.  An
    unknown or unclassified reference remains countable and therefore fails
    closed when a metric is partially declared.
    """

    rows = tuple(operations)
    total_cases = sum(_case_bound(row) for row in rows)
    issues: list[ResourceAccountingIssue] = []

    cpu_time_s, metric_issues = _operation_metric(
        rows,
        bound_name="max_cpu_time_s",
        metric_name="cpu_time_s",
        noun="CPU-time",
    )
    issues.extend(metric_issues)

    work_units, metric_issues = _operation_metric(
        rows,
        bound_name="max_work_units",
        metric_name="work_units",
        noun="work",
    )
    issues.extend(metric_issues)
    if work_units is None and "max_work_units" in envelope and not any(
        "max_work_units" in dict(row.get("bounds", {}) or {}) for row in rows
    ):
        issues.append(
            ResourceAccountingIssue(
                "resource.work_units_estimate_missing",
                "A max_work_units plan cap requires a declared work estimate for every operation.",
                ("$.plan.resource_envelope.max_work_units", "$.plan.operations"),
                {"metric": "work_units", "maximum": envelope["max_work_units"]},
                "Declare bounds.max_work_units for every operation before setting a work cap.",
            )
        )

    typed_metrics: dict[str, int | None] = {}
    for metric_name, bound_name, noun in (
        ("propagation_steps", "max_propagation_steps", "propagation-step"),
        ("estimator_evaluations", "max_estimator_evaluations", "estimator-evaluation"),
        ("batch_evaluations", "max_batch_evaluations", "batch-evaluation"),
    ):
        metric_value, metric_issues = _operation_integer_metric(
            rows,
            bound_name=bound_name,
            metric_name=metric_name,
            noun=noun,
        )
        issues.extend(metric_issues)
        typed_metrics[metric_name] = metric_value
        if metric_value is None and bound_name in envelope and not any(
            bound_name in dict(row.get("bounds", {}) or {}) for row in rows
        ):
            issues.append(
                ResourceAccountingIssue(
                    f"resource.{metric_name}_estimate_missing",
                    f"A {bound_name} plan cap requires a declared estimate for every operation.",
                    (f"$.plan.resource_envelope.{bound_name}", "$.plan.operations"),
                    {"metric": metric_name, "maximum": envelope[bound_name]},
                    f"Declare bounds.{bound_name} for every operation before setting this cap.",
                )
            )

    arc_values, arc_issues = _operation_metric_values(
        rows,
        bound_name="max_arc_duration_s",
        metric_name="arc_duration_s",
        noun="arc-duration",
    )
    issues.extend(arc_issues)
    arc_duration_s = None if arc_values is None else max(arc_values.values(), default=0.0)
    if arc_duration_s is None and "max_arc_duration_s" in envelope and not any(
        "max_arc_duration_s" in dict(row.get("bounds", {}) or {}) for row in rows
    ):
        issues.append(
            ResourceAccountingIssue(
                "resource.arc_duration_s_estimate_missing",
                "A max_arc_duration_s plan cap requires a per-operation arc-duration estimate.",
                ("$.plan.resource_envelope.max_arc_duration_s", "$.plan.operations"),
                {"metric": "max_arc_duration_s", "maximum": envelope["max_arc_duration_s"]},
                "Declare bounds.max_arc_duration_s for every operation before setting an arc-duration cap.",
            )
        )

    memory_by_operation, metric_issues = _operation_metric_values(
        rows,
        bound_name="max_peak_memory_mb",
        metric_name="peak_memory_mb",
        noun="peak-memory",
    )
    issues.extend(metric_issues)
    worker_bound, worker_issues = _worker_bound(envelope, len(rows))
    issues.extend(worker_issues)
    peak_memory_mb: float | None = None
    if memory_by_operation is not None:
        peak_memory_mb, schedule_issues = _scheduled_peak_memory(
            rows,
            memory_by_operation,
            worker_bound,
        )
        issues.extend(schedule_issues)

    unique_input_bytes, input_issues = _unique_input_bytes(rows)
    issues.extend(input_issues)
    if unique_input_bytes is None and "max_input_bytes" in envelope and not any(
        _INPUT_BYTES_KEY in metadata
        for _, _, _, _ref, metadata in _external_input_metadata_rows(rows)
    ):
        issues.append(
            ResourceAccountingIssue(
                "resource.input_bytes_estimate_missing",
                "A max_input_bytes plan cap requires declared byte estimates for external inputs.",
                ("$.plan.resource_envelope.max_input_bytes", "$.plan.operations"),
                {"metric": "unique_input_bytes", "maximum": envelope["max_input_bytes"]},
                "Declare metadata.input_bytes and metadata.content_sha256 for every external input before setting an input cap.",
            )
        )

    retained_output_bytes, output_issues = _retained_output_bytes(rows)
    issues.extend(output_issues)
    if retained_output_bytes is None and "max_retained_output_bytes" in envelope and not any(
        _has_retained_metadata(dict(ref.get("metadata", {}) or {}))
        for row in rows
        for ref in list(row.get("output_refs", []) or [])
    ):
        issues.append(
            ResourceAccountingIssue(
                "resource.retained_output_estimate_missing",
                "A max_retained_output_bytes plan cap requires declared retained sizes for outputs.",
                ("$.plan.resource_envelope.max_retained_output_bytes", "$.plan.operations"),
                {"metric": "retained_output_bytes", "maximum": envelope["max_retained_output_bytes"]},
                "Declare metadata.retained_bytes for every output before setting a retained-output cap.",
            )
        )

    observation_count, observation_issues = _unique_input_count(
        rows,
        metric_name="observation_count",
        metadata_keys=_OBSERVATION_COUNT_KEYS,
        envelope=envelope,
        envelope_key="max_observations",
        eligible_refs=(
            None
            if countable_input_refs is None
            else countable_input_refs.get("observation_count", set())
        ),
    )
    issues.extend(observation_issues)
    station_count, station_issues = _unique_input_count(
        rows,
        metric_name="station_count",
        metadata_keys=_STATION_COUNT_KEYS,
        envelope=envelope,
        envelope_key="max_stations",
        eligible_refs=(
            None
            if countable_input_refs is None
            else countable_input_refs.get("station_count", set())
        ),
    )
    issues.extend(station_issues)

    return ResourceAccounting(
        total_cases=total_cases,
        cpu_time_s=cpu_time_s,
        work_units=work_units,
        propagation_steps=typed_metrics["propagation_steps"],
        estimator_evaluations=typed_metrics["estimator_evaluations"],
        batch_evaluations=typed_metrics["batch_evaluations"],
        arc_duration_s=arc_duration_s,
        peak_memory_mb=peak_memory_mb,
        unique_input_bytes=unique_input_bytes,
        retained_output_bytes=retained_output_bytes,
        observation_count=observation_count,
        station_count=station_count,
        worker_bound=worker_bound,
        issues=tuple(issues),
    )


def _case_bound(operation: Mapping[str, Any]) -> int:
    value = dict(operation.get("bounds", {}) or {}).get("max_cases", 1)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _operation_metric(
    operations: Sequence[Mapping[str, Any]],
    *,
    bound_name: str,
    metric_name: str,
    noun: str,
) -> tuple[float | None, list[ResourceAccountingIssue]]:
    values, issues = _operation_metric_values(
        operations,
        bound_name=bound_name,
        metric_name=metric_name,
        noun=noun,
    )
    if values is None:
        return None, issues
    return sum(values.values()), issues


def _operation_integer_metric(
    operations: Sequence[Mapping[str, Any]],
    *,
    bound_name: str,
    metric_name: str,
    noun: str,
) -> tuple[int | None, list[ResourceAccountingIssue]]:
    values, issues = _operation_metric_values(
        operations,
        bound_name=bound_name,
        metric_name=metric_name,
        noun=noun,
    )
    if values is None:
        return None, issues
    if any(not value.is_integer() for value in values.values()):
        for operation_id, value in values.items():
            if not value.is_integer():
                issues.append(
                    ResourceAccountingIssue(
                        f"resource.{metric_name}_estimate_invalid",
                        f"Operation {operation_id!r} has a non-integer {noun} estimate.",
                        (f"$.plan.operations[{operation_id}].bounds.{bound_name}",),
                        {"metric": metric_name, "operation_id": operation_id, "value": value},
                        f"Declare a non-negative integer bounds.{bound_name} value.",
                    )
                )
        return None, issues
    return int(sum(values.values())), issues


def _operation_metric_values(
    operations: Sequence[Mapping[str, Any]],
    *,
    bound_name: str,
    metric_name: str,
    noun: str,
) -> tuple[dict[str, float] | None, list[ResourceAccountingIssue]]:
    bounds_by_operation = {
        str(operation.get("operation_id", index)): dict(operation.get("bounds", {}) or {})
        for index, operation in enumerate(operations)
    }
    if not any(bound_name in bounds for bounds in bounds_by_operation.values()):
        return None, []

    issues: list[ResourceAccountingIssue] = []
    missing = [
        operation_id
        for operation_id, bounds in bounds_by_operation.items()
        if bound_name not in bounds
    ]
    if missing:
        issues.append(
            ResourceAccountingIssue(
                f"resource.{metric_name}_estimate_missing",
                f"Metric-specific {noun} estimates must be declared for every operation once enabled.",
                ("$.plan.operations",),
                {"metric": metric_name, "missing_operation_ids": sorted(missing)},
                f"Declare bounds.{bound_name} for every operation; do not infer it from another metric.",
            )
        )

    values: dict[str, float] = {}
    for operation_id, bounds in bounds_by_operation.items():
        if bound_name not in bounds:
            continue
        value = bounds[bound_name]
        numeric = _finite_nonnegative(value)
        if numeric is None:
            issues.append(
                ResourceAccountingIssue(
                    f"resource.{metric_name}_estimate_invalid",
                    f"Operation {operation_id!r} has an invalid {noun} estimate.",
                    (f"$.plan.operations[{operation_id}].bounds.{bound_name}",),
                    {"metric": metric_name, "operation_id": operation_id, "value": value},
                    f"Declare a finite non-negative bounds.{bound_name} value.",
                )
            )
            continue
        values[operation_id] = numeric

    if missing or any(issue.code.endswith("_invalid") for issue in issues):
        return None, issues
    return values, issues


def _unique_input_bytes(
    operations: Sequence[Mapping[str, Any]],
) -> tuple[int | None, list[ResourceAccountingIssue]]:
    metadata_rows = _external_input_metadata_rows(operations)
    if not any(_INPUT_BYTES_KEY in metadata for _, _, _, _, metadata in metadata_rows):
        return None, []

    issues: list[ResourceAccountingIssue] = []
    content_sizes: dict[str, int] = {}
    for _operation_index, operation_id, ref_index, ref, metadata in metadata_rows:
        path = f"$.plan.operations[{operation_id}].input_refs[{ref_index}].metadata"
        if _INPUT_BYTES_KEY not in metadata:
            issues.append(
                ResourceAccountingIssue(
                    "resource.input_bytes_estimate_missing",
                    "Unique external input-byte accounting requires a byte estimate for every external input reference.",
                    (path,),
                    {"operation_id": operation_id, "ref_id": ref.get("ref_id")},
                    "Declare metadata.input_bytes for every external input reference or remove byte accounting.",
                )
            )
            continue

        byte_count = _nonnegative_integer(metadata[_INPUT_BYTES_KEY])
        if byte_count is None:
            issues.append(
                ResourceAccountingIssue(
                    "resource.input_bytes_estimate_invalid",
                    "Input-byte metadata must be a finite non-negative integer.",
                    (path,),
                    {
                        "operation_id": operation_id,
                        "ref_id": ref.get("ref_id"),
                        "value": metadata[_INPUT_BYTES_KEY],
                    },
                    "Use the exact measured byte count as metadata.input_bytes.",
                )
            )
            continue

        digest = metadata.get("content_sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            issues.append(
                ResourceAccountingIssue(
                    "resource.input_content_identity_missing",
                    "Unique external input-byte accounting requires a valid content_sha256 for every external input.",
                    (path,),
                    {
                        "operation_id": operation_id,
                        "ref_id": ref.get("ref_id"),
                        "content_sha256": digest,
                    },
                    "Bind each external input to its exact lowercase SHA-256 content digest.",
                )
            )
            continue

        previous = content_sizes.get(digest)
        if previous is not None and previous != byte_count:
            issues.append(
                ResourceAccountingIssue(
                    "resource.input_content_size_conflict",
                    "One external content digest has conflicting input-byte estimates.",
                    (path,),
                    {
                        "content_sha256": digest,
                        "previous_bytes": previous,
                        "observed_bytes": byte_count,
                    },
                    "Use one measured byte count for each content digest.",
                )
            )
            continue
        content_sizes[digest] = byte_count

    if issues:
        return None, issues
    return sum(content_sizes.values()), issues


def _unique_input_count(
    operations: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    metadata_keys: Sequence[str],
    envelope: Mapping[str, Any],
    envelope_key: str,
    eligible_refs: set[tuple[int, int]] | None,
) -> tuple[int | None, list[ResourceAccountingIssue]]:
    """Account a plan-wide input count using the documented identity rules.

    A valid lowercase SHA-256 digest identifies one content item across all
    references, so repeated references contribute once.  A reference without
    a resolvable digest remains distinct and contributes its declared count on
    each occurrence.  Once a metric is opted into, every input reference must
    provide one of its accepted metadata spellings; this keeps partial counts
    from being presented as assessed totals.  A cap with no count metadata is
    likewise unassessed and therefore blocks preflight.
    """

    rows = [
        row
        for row in _all_input_metadata_rows(operations)
        if eligible_refs is None or (row[0], row[2]) in eligible_refs
    ]
    cap_declared = envelope_key in envelope
    has_count_metadata = any(
        any(key in metadata for key in metadata_keys)
        for _, _, _, _, metadata in rows
    )
    if not has_count_metadata:
        if not cap_declared:
            return None, []
        return None, [
            ResourceAccountingIssue(
                f"resource.{metric_name}_estimate_missing",
                f"A {envelope_key} plan cap requires declared count metadata for every input reference.",
                (f"$.plan.resource_envelope.{envelope_key}", "$.plan.operations"),
                {"metric": metric_name, "maximum": envelope[envelope_key]},
                f"Declare one of {', '.join(metadata_keys)} for every input reference before setting this cap.",
            )
        ]

    issues: list[ResourceAccountingIssue] = []
    values_by_identity: dict[tuple[Any, ...], list[tuple[int | None, str]]] = {}
    missing_paths: list[str] = []
    for operation_index, operation_id, ref_index, ref, metadata in rows:
        path = f"$.plan.operations[{operation_id}].input_refs[{ref_index}].metadata"
        digest = metadata.get("content_sha256")
        if isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None:
            identity: tuple[Any, ...] = ("digest", digest)
        else:
            identity = ("reference", operation_index, ref_index)

        raw_values = [(key, metadata[key]) for key in metadata_keys if key in metadata]
        if not raw_values:
            missing_paths.append(path)
            values_by_identity.setdefault(identity, []).append((None, path))
            continue

        parsed_values: list[int] = []
        invalid_values: list[tuple[str, Any]] = []
        for key, value in raw_values:
            parsed = _nonnegative_integer(value)
            if parsed is None:
                invalid_values.append((key, value))
            else:
                parsed_values.append(parsed)
        if invalid_values:
            issues.append(
                ResourceAccountingIssue(
                    f"resource.{metric_name}_estimate_invalid",
                    f"Input {metric_name} metadata must be a finite non-negative integer.",
                    (path,),
                    {
                        "metric": metric_name,
                        "operation_id": operation_id,
                        "ref_id": ref.get("ref_id"),
                        "values": {key: value for key, value in invalid_values},
                    },
                    f"Use one finite non-negative integer among {', '.join(metadata_keys)} for each input reference.",
                )
            )
            values_by_identity.setdefault(identity, []).append((None, path))
            continue

        if len(set(parsed_values)) != 1:
            issues.append(
                ResourceAccountingIssue(
                    f"resource.{metric_name}_estimate_conflict",
                    f"Input metadata contains conflicting {metric_name} aliases.",
                    (path,),
                    {
                        "metric": metric_name,
                        "operation_id": operation_id,
                        "ref_id": ref.get("ref_id"),
                        "values": {key: value for key, value in raw_values},
                    },
                    f"Use one matching value among {', '.join(metadata_keys)} for each input reference.",
                )
            )
            values_by_identity.setdefault(identity, []).append((None, path))
            continue

        values_by_identity.setdefault(identity, []).append((parsed_values[0], path))

    if missing_paths:
        issues.extend(
            ResourceAccountingIssue(
                f"resource.{metric_name}_estimate_missing",
                f"Metric-specific {metric_name} estimates must be declared for every input reference once enabled.",
                (path,),
                {"metric": metric_name},
                f"Declare one of {', '.join(metadata_keys)} for every input reference or remove {metric_name} accounting.",
            )
            for path in missing_paths
        )

    totals: list[int] = []
    for identity, entries in values_by_identity.items():
        known_values = [value for value, _path in entries if value is not None]
        if not known_values:
            continue
        if len(set(known_values)) != 1:
            paths = tuple(path for _value, path in entries)
            issues.append(
                ResourceAccountingIssue(
                    f"resource.{metric_name}_estimate_conflict",
                    f"One content identity has conflicting {metric_name} estimates.",
                    paths,
                    {
                        "metric": metric_name,
                        "content_sha256": identity[1]
                        if identity and identity[0] == "digest"
                        else None,
                        "values": known_values,
                    },
                    "Use one measured count for each content identity.",
                )
            )
            continue
        # Digest identities are deduplicated by the dictionary key.  Unresolved
        # references each have their own key and therefore remain conservative.
        totals.append(known_values[0])

    if issues:
        return None, issues
    return sum(totals), []


def _all_input_metadata_rows(
    operations: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str, int, Mapping[str, Any], dict[str, Any]]]:
    rows: list[tuple[int, str, int, Mapping[str, Any], dict[str, Any]]] = []
    for operation_index, operation in enumerate(operations):
        operation_id = str(operation.get("operation_id", operation_index))
        for ref_index, ref in enumerate(list(operation.get("input_refs", []) or [])):
            raw_metadata = ref.get("metadata", {})
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            rows.append((operation_index, operation_id, ref_index, ref, metadata))
    return rows


def _external_input_metadata_rows(
    operations: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str, int, Mapping[str, Any], dict[str, Any]]]:
    derived_keys = _authenticated_derived_input_keys(operations)
    rows: list[tuple[int, str, int, Mapping[str, Any], dict[str, Any]]] = []
    for operation_index, operation in enumerate(operations):
        operation_id = str(operation.get("operation_id", operation_index))
        for ref_index, ref in enumerate(list(operation.get("input_refs", []) or [])):
            if (operation_index, ref_index) in derived_keys:
                continue
            rows.append(
                (
                    operation_index,
                    operation_id,
                    ref_index,
                    ref,
                    dict(ref.get("metadata", {}) or {}),
                )
            )
    return rows


def _authenticated_derived_input_keys(
    operations: Sequence[Mapping[str, Any]],
) -> set[tuple[int, int]]:
    """Return input refs proven to be direct copies of producer outputs.

    Resource accounting is deliberately fail-closed here.  A producer claim in
    input metadata is insufficient on its own: the operation must declare a
    unique operation identity, a direct dependency edge, and a unique output
    reference whose kind, schema, producer, and semantic role match the input.
    References that fail any check remain external and therefore still need
    byte and content-identity metadata.
    """

    operation_rows = [
        (operation_index, operation)
        for operation_index, operation in enumerate(operations)
    ]
    operation_indices_by_id: dict[str, list[int]] = {}
    outputs_by_ref_id: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for operation_index, operation in operation_rows:
        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            continue
        operation_indices_by_id.setdefault(operation_id, []).append(operation_index)
        for output_ref in list(operation.get("output_refs", []) or []):
            ref_id = output_ref.get("ref_id")
            if isinstance(ref_id, str) and ref_id:
                outputs_by_ref_id.setdefault(ref_id, []).append((operation_id, output_ref))

    derived_keys: set[tuple[int, int]] = set()
    for operation_index, operation in operation_rows:
        operation_id = operation.get("operation_id")
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or len(operation_indices_by_id.get(operation_id, [])) != 1
        ):
            continue
        depends_on = operation.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            not isinstance(item, str) for item in depends_on
        ) or len(depends_on) != len(set(depends_on)):
            continue
        dependency_ids = set(depends_on)
        for ref_index, input_ref in enumerate(list(operation.get("input_refs", []) or [])):
            metadata = dict(input_ref.get("metadata", {}) or {})
            producer_id = metadata.get("producer_operation_id")
            semantic_role = metadata.get("semantic_role")
            if (
                not isinstance(producer_id, str)
                or not producer_id
                or not isinstance(semantic_role, str)
                or not semantic_role
                or producer_id == operation_id
                or producer_id not in dependency_ids
                or len(operation_indices_by_id.get(producer_id, [])) != 1
            ):
                continue

            ref_id = input_ref.get("ref_id")
            candidates = outputs_by_ref_id.get(ref_id, []) if isinstance(ref_id, str) else []
            if len(candidates) != 1:
                continue
            output_producer_id, output_ref = candidates[0]
            output_metadata = dict(output_ref.get("metadata", {}) or {})
            if (
                output_producer_id != producer_id
                or output_metadata.get("producer_operation_id") != producer_id
                or output_metadata.get("semantic_role") != semantic_role
                or not isinstance(input_ref.get("kind"), str)
                or not input_ref.get("kind")
                or input_ref.get("kind") != output_ref.get("kind")
                or not isinstance(output_ref.get("kind"), str)
                or not output_ref.get("kind")
                or not isinstance(input_ref.get("schema_id"), str)
                or not input_ref.get("schema_id")
                or input_ref.get("schema_id") != output_ref.get("schema_id")
                or not isinstance(output_ref.get("schema_id"), str)
                or not output_ref.get("schema_id")
            ):
                continue
            derived_keys.add((operation_index, ref_index))
    return derived_keys


def _retained_output_bytes(
    operations: Sequence[Mapping[str, Any]],
) -> tuple[int | None, list[ResourceAccountingIssue]]:
    refs = [
        (str(operation.get("operation_id", operation_index)), ref_index, operation, ref)
        for operation_index, operation in enumerate(operations)
        for ref_index, ref in enumerate(list(operation.get("output_refs", []) or []))
    ]
    metadata_rows = [
        (operation_id, ref_index, operation, ref, dict(ref.get("metadata", {}) or {}))
        for operation_id, ref_index, operation, ref in refs
    ]
    if not any(_has_retained_metadata(metadata) for _, _, _, _, metadata in metadata_rows):
        return None, []

    issues: list[ResourceAccountingIssue] = []
    totals_by_operation: dict[str, int] = {}
    for operation_id, ref_index, _operation, ref, metadata in metadata_rows:
        path = f"$.plan.operations[{operation_id}].output_refs[{ref_index}].metadata"
        value = _retained_value(metadata)
        if value is None:
            issues.append(
                ResourceAccountingIssue(
                    "resource.retained_output_estimate_missing",
                    "Retained-output accounting requires a declared size for every output reference.",
                    (path,),
                    {"operation_id": operation_id, "ref_id": ref.get("ref_id")},
                    "Declare metadata.retained_bytes for every output reference, including zero for omitted output.",
                )
            )
            continue
        byte_count = _nonnegative_integer(value)
        if byte_count is None:
            issues.append(
                ResourceAccountingIssue(
                    "resource.retained_output_estimate_invalid",
                    "Retained-output metadata must be a finite non-negative integer.",
                    (path,),
                    {
                        "operation_id": operation_id,
                        "ref_id": ref.get("ref_id"),
                        "retained_bytes": value,
                    },
                    "Use the measured retained byte count in metadata.retained_bytes.",
                )
            )
            continue
        totals_by_operation[operation_id] = totals_by_operation.get(operation_id, 0) + byte_count

    if issues:
        return None, issues

    for operation_id, total in totals_by_operation.items():
        operation = next(
            row
            for row in operations
            if str(row.get("operation_id", "")) == operation_id
        )
        bounds = dict(operation.get("bounds", {}) or {})
        declared = bounds.get("max_retained_output_bytes")
        if declared is None:
            issues.append(
                ResourceAccountingIssue(
                    "resource.retained_output_bound_missing",
                    "Each operation with retained-output accounting needs a per-operation output bound.",
                    (f"$.plan.operations[{operation_id}].bounds.max_retained_output_bytes",),
                    {"operation_id": operation_id, "observed_retained_bytes": total},
                    "Declare max_retained_output_bytes at least as large as the retained output total.",
                )
            )
            continue
        declared_int = _nonnegative_integer(declared)
        if declared_int is None or total > declared_int:
            issues.append(
                ResourceAccountingIssue(
                    "resource.retained_output_bound_insufficient",
                    f"Operation {operation_id!r} underbounds its retained output total.",
                    (f"$.plan.operations[{operation_id}].bounds.max_retained_output_bytes",),
                    {
                        "operation_id": operation_id,
                        "declared": declared,
                        "required_minimum": total,
                    },
                    "Increase max_retained_output_bytes to the measured retained total.",
                )
            )
    if issues:
        return None, issues
    return sum(totals_by_operation.values()), issues


def _retained_value(metadata: Mapping[str, Any]) -> Any:
    for key in _RETAINED_BYTES_KEYS:
        if key in metadata:
            return metadata[key]
    if metadata.get("retained") is True and "output_bytes" in metadata:
        return metadata["output_bytes"]
    if metadata.get("retained") is False and "output_bytes" in metadata:
        return 0
    return None


def _has_retained_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(key in metadata for key in _RETAINED_BYTES_KEYS) or (
        "retained" in metadata and "output_bytes" in metadata
    )


def _scheduled_peak_memory(
    operations: Sequence[Mapping[str, Any]],
    memory_by_operation: Mapping[str, float],
    worker_bound: int,
) -> tuple[float | None, list[ResourceAccountingIssue]]:
    """Return a safe upper bound for dependency-respecting concurrency.

    Operation wall-time bounds do not determine actual overlap: a task may
    finish much earlier than its declared maximum and release a worker for a
    dependent task while another independent task is still running.  We
    therefore bound every legal concurrent set directly.  Any such set is an
    antichain of the dependency graph, so for each possible anchor we add its
    memory to the largest ``worker_bound - 1`` memories that are incomparable
    with it.  The bound is conservative for mixed DAGs because those selected
    incomparable operations need not be mutually incomparable.
    """
    operation_ids = [
        str(operation.get("operation_id", index))
        for index, operation in enumerate(operations)
    ]
    operation_by_id = {
        str(operation.get("operation_id", index)): operation
        for index, operation in enumerate(operations)
    }
    dependencies = {
        operation_id: {str(value) for value in operation_by_id[operation_id].get("depends_on", [])}
        for operation_id in operation_ids
    }
    unknown = sorted(
        {
            dependency
            for dependency_set in dependencies.values()
            for dependency in dependency_set
            if dependency not in operation_by_id
        }
    )
    if unknown:
        return None, [
            ResourceAccountingIssue(
                "resource.memory_dependency_invalid",
                "Peak-memory scheduling cannot resolve an unknown dependency.",
                ("$.plan.operations",),
                {"dependency_ids": unknown},
                "Use only operation identifiers declared by this plan.",
            )
        ]

    ancestors: dict[str, set[str]] = {}
    visiting: set[str] = set()

    def collect_ancestors(operation_id: str) -> set[str]:
        if operation_id in ancestors:
            return ancestors[operation_id]
        if operation_id in visiting:
            raise ValueError(operation_id)
        visiting.add(operation_id)
        found: set[str] = set()
        for dependency in dependencies[operation_id]:
            found.add(dependency)
            found.update(collect_ancestors(dependency))
        visiting.remove(operation_id)
        ancestors[operation_id] = found
        return found

    try:
        for operation_id in operation_ids:
            collect_ancestors(operation_id)
    except ValueError:
        cycle_members = sorted(operation_id for operation_id in visiting)
        return None, [
            ResourceAccountingIssue(
                "resource.memory_dependency_cycle",
                "Peak-memory accounting cannot resolve a dependency cycle.",
                ("$.plan.operations",),
                {"operation_ids": cycle_members},
                "Remove dependency cycles before requesting resource accounting.",
            )
        ]

    peak_memory = 0.0
    for anchor in operation_ids:
        incomparable = [
            operation_id
            for operation_id in operation_ids
            if operation_id != anchor
            and operation_id not in ancestors[anchor]
            and anchor not in ancestors[operation_id]
        ]
        largest_incomparable = sorted(
            (float(memory_by_operation[operation_id]) for operation_id in incomparable),
            reverse=True,
        )[: max(worker_bound - 1, 0)]
        peak_memory = max(peak_memory, float(memory_by_operation[anchor]) + sum(largest_incomparable))

    return peak_memory, []


def _worker_bound(
    envelope: Mapping[str, Any],
    operation_count: int,
) -> tuple[int, list[ResourceAccountingIssue]]:
    value = envelope.get("max_parallel_workers", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1, [
            ResourceAccountingIssue(
                "resource.worker_bound_invalid",
                "The plan worker bound must be a positive integer.",
                ("$.plan.resource_envelope.max_parallel_workers",),
                {"value": value},
                "Declare a positive integer max_parallel_workers.",
            )
        ]
    return max(1, min(value, max(operation_count, 1))), []


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    return numeric


def _nonnegative_integer(value: Any) -> int | None:
    numeric = _finite_nonnegative(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


__all__ = ["ResourceAccounting", "ResourceAccountingIssue", "account_plan_resources"]
