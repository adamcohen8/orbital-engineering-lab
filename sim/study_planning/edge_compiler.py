"""Deterministically compile typed artifact edges for model-proposed operation graphs."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .capabilities import CapabilityCatalog, CapabilityPort
from .contracts import canonical_sha256

EDGE_COMPILATION_SCHEMA = "oel.study_edge_compilation.v1"


@dataclass(frozen=True, slots=True)
class EdgeCompilationFinding:
    code: str
    message: str
    paths: tuple[str, ...]
    observed: Mapping[str, Any]
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": "blocker",
            "category": "planner_error",
            "message": self.message,
            "paths": list(self.paths),
            "observed": dict(self.observed),
            "suggestion": self.suggestion,
        }


class StudyEdgeCompilationError(ValueError):
    def __init__(self, findings: Sequence[EdgeCompilationFinding]) -> None:
        self.findings = tuple(findings)
        detail = "; ".join(f"{item.code}: {item.message}" for item in self.findings[:8])
        super().__init__(detail or "Study operation edges could not be compiled.")


def compile_typed_operation_edges(
    operations: Sequence[Mapping[str, Any]],
    *,
    request_inputs: Sequence[Mapping[str, Any]],
    catalog: CapabilityCatalog,
) -> dict[str, Any]:
    """Add canonical input/output references using capability type contracts.

    Models declare operations, direct dependencies, parameters, and bounds. The
    host selects only type-compatible request inputs or direct dependency
    outputs. Multiple compatible sources of the same kind are rejected as
    ambiguous instead of being chosen heuristically.
    """

    proposed = [deepcopy(dict(item)) for item in operations]
    findings = _graph_findings(proposed)
    if findings:
        raise StudyEdgeCompilationError(findings)

    operation_by_id = {str(item["operation_id"]): item for item in proposed}
    ordered_ids = _topological_order(proposed)
    request_sources = [_request_source(item) for item in request_inputs]
    outputs_by_operation: dict[str, list[dict[str, Any]]] = {}
    compiled_by_id: dict[str, dict[str, Any]] = {}
    for operation_id in ordered_ids:
        operation = operation_by_id[operation_id]
        capability_id = str(operation["capability_id"])
        capability = catalog.get(capability_id)
        if capability is None:
            raise StudyEdgeCompilationError(
                (
                EdgeCompilationFinding(
                    "edge.capability_unavailable",
                    f"Capability {capability_id!r} is not present in the authoritative catalog.",
                    (f"$.plan.operations[{operation_id}].capability_id",),
                    {"capability_id": capability_id},
                    "Select a capability from the supplied authoritative offer.",
                )
                ,)
            )

        dependency_refs: list[dict[str, Any]] = [
            ref
            for dependency_id in operation["depends_on"]
            for ref in outputs_by_operation.get(str(dependency_id), [])
        ]
        input_refs: list[dict[str, Any]] = []
        operation_findings: list[EdgeCompilationFinding] = []
        for port in capability.compiled_input_ports:
            dependency_matches = [
                ref for ref in dependency_refs if _matches_port(ref, port)
            ]
            request_matches = [ref for ref in request_sources if _matches_port(ref, port)]
            matches = dependency_matches or request_matches
            if port.cardinality in {"exactly_one", "optional"} and len(matches) > 1:
                operation_findings.append(
                    EdgeCompilationFinding(
                        "edge.input_source_ambiguous",
                        (
                            f"Operation {operation_id!r} has multiple direct sources for "
                            f"named port {port.port_id!r}."
                        ),
                        (f"$.plan.operations[{operation_id}].depends_on",),
                        {
                            "operation_id": operation_id,
                            "port_id": port.port_id,
                            "semantic_role": port.semantic_role,
                            "accepted_schema_ids": list(port.schema_ids),
                            "candidate_ref_ids": [item["ref_id"] for item in matches],
                        },
                        "Narrow the direct dependencies so each named input port has one source.",
                    )
                )
            elif port.cardinality in {"exactly_one", "many"} and not matches:
                operation_findings.append(
                    EdgeCompilationFinding(
                        "edge.required_port_missing",
                        f"Operation {operation_id!r} has no compatible source for required port {port.port_id!r}.",
                        (f"$.plan.operations[{operation_id}].depends_on",),
                        {
                            "operation_id": operation_id,
                            "port_id": port.port_id,
                            "semantic_role": port.semantic_role,
                            "artifact_kind": port.artifact_kind,
                            "accepted_schema_ids": list(port.schema_ids),
                            "required_metadata": list(port.required_metadata),
                        },
                        "Supply an identity-bound artifact matching the named port contract.",
                    )
                )
            else:
                selected = matches if port.cardinality == "many" else matches[:1]
                input_refs.extend(_bind_input_ref(item, port.port_id) for item in selected)

        operation_findings.extend(
            _resource_bound_findings(operation_id, operation.get("bounds", {}), input_refs)
        )

        available_refs = dependency_refs or request_sources
        if (
            not operation_findings
            and capability.compiled_input_ports
            and available_refs
            and not input_refs
        ):
            operation_findings.append(
                EdgeCompilationFinding(
                    "edge.no_compatible_input",
                    f"Operation {operation_id!r} has no type-compatible direct input source.",
                    (f"$.plan.operations[{operation_id}].depends_on",),
                    {
                        "operation_id": operation_id,
                        "accepted_ports": [item.to_dict() for item in capability.compiled_input_ports],
                        "available_kinds": sorted({item["kind"] for item in available_refs}),
                    },
                    "Add a dependency that produces one of the capability's accepted input kinds.",
                )
            )
        if operation_findings:
            raise StudyEdgeCompilationError(operation_findings)

        output_refs = [
            {
                "ref_id": f"{operation_id}.{port.port_id}",
                "kind": port.artifact_kind,
                "port_id": port.port_id,
                "schema_id": port.schema_ids[0],
                "metadata": {
                    "producer_operation_id": operation_id,
                    "semantic_role": port.semantic_role,
                },
            }
            for port in capability.compiled_output_ports
        ]
        compiled = deepcopy(operation)
        compiled["input_refs"] = input_refs
        compiled["output_refs"] = output_refs
        compiled_by_id[operation_id] = compiled
        outputs_by_operation[operation_id] = output_refs

    compiled_operations = [compiled_by_id[operation_id] for operation_id in ordered_ids]
    receipt = {
        "schema": EDGE_COMPILATION_SCHEMA,
        "catalog_scope": catalog.scope,
        "catalog_sha256": canonical_sha256(catalog.to_list()),
        "proposed_operation_sha256": canonical_sha256(proposed),
        "request_input_sha256": canonical_sha256(list(request_inputs)),
        "compiled_operation_sha256": canonical_sha256(compiled_operations),
        "operation_count": len(compiled_operations),
        "edge_count": sum(len(item["input_refs"]) for item in compiled_operations),
        "selection_policy": "direct_dependency_then_request_input_by_named_port_schema_and_metadata",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return {"operations": compiled_operations, "receipt": receipt}


def verify_typed_operation_edges(
    operations: Sequence[Mapping[str, Any]],
    *,
    request_inputs: Sequence[Mapping[str, Any]],
    catalog: CapabilityCatalog,
) -> dict[str, Any]:
    """Verify that a bound plan preserves every authoritative named-port contract.

    Unlike :func:`compile_typed_operation_edges`, this accepts the plan's
    content-bound reference identifiers.  It requires the same direct-source,
    schema, metadata, cardinality, and resource-bound semantics and therefore
    closes the gap between model-harness compilation and hosted execution.
    """

    proposed = [deepcopy(dict(item)) for item in operations]
    graph_findings = _graph_findings(proposed)
    if graph_findings:
        raise StudyEdgeCompilationError(graph_findings)

    operation_by_id = {str(item["operation_id"]): item for item in proposed}
    request_sources = {
        str(item["ref_id"]): item for item in (_request_source(row) for row in request_inputs)
    }
    findings: list[EdgeCompilationFinding] = []
    for operation_id in _topological_order(proposed):
        operation = operation_by_id[operation_id]
        capability_id = str(operation["capability_id"])
        capability = catalog.get(capability_id)
        if capability is None:
            findings.append(
                EdgeCompilationFinding(
                    "edge.capability_unavailable",
                    f"Capability {capability_id!r} is not present in the authoritative catalog.",
                    (f"$.plan.operations[{operation_id}].capability_id",),
                    {"capability_id": capability_id},
                    "Select a capability from the supplied authoritative offer.",
                )
            )
            continue

        dependency_sources = {
            str(ref["ref_id"]): ref
            for dependency_id in operation["depends_on"]
            for ref in operation_by_id[str(dependency_id)]["output_refs"]
        }
        available_sources = {**request_sources, **dependency_sources}
        input_refs = [deepcopy(dict(item)) for item in operation["input_refs"]]
        output_refs = [deepcopy(dict(item)) for item in operation["output_refs"]]
        findings.extend(
            _verify_port_references(
                operation_id,
                input_refs,
                capability.compiled_input_ports,
                available_sources=available_sources,
                direction="input",
            )
        )
        findings.extend(
            _verify_port_references(
                operation_id,
                output_refs,
                capability.compiled_output_ports,
                available_sources=None,
                direction="output",
            )
        )
        findings.extend(
            _resource_bound_findings(operation_id, operation.get("bounds", {}), input_refs)
        )

    if findings:
        raise StudyEdgeCompilationError(findings)
    receipt = {
        "schema": "oel.study_edge_verification.v1",
        "catalog_scope": catalog.scope,
        "catalog_sha256": canonical_sha256(catalog.to_list()),
        "request_input_sha256": canonical_sha256(list(request_inputs)),
        "compiled_operation_sha256": canonical_sha256(proposed),
        "operation_count": len(proposed),
        "edge_count": sum(len(item["input_refs"]) for item in proposed),
        "verification_policy": "direct_dependency_or_request_source_by_named_port_schema_and_exact_metadata",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _verify_port_references(
    operation_id: str,
    references: Sequence[Mapping[str, Any]],
    ports: Sequence[CapabilityPort],
    *,
    available_sources: Mapping[str, Mapping[str, Any]] | None,
    direction: str,
) -> list[EdgeCompilationFinding]:
    findings: list[EdgeCompilationFinding] = []
    port_by_id = {port.port_id: port for port in ports}
    grouped: dict[str, list[Mapping[str, Any]]] = {port.port_id: [] for port in ports}
    for reference in references:
        port_id = str(reference.get("port_id", ""))
        port = port_by_id.get(port_id)
        if port is None:
            findings.append(
                EdgeCompilationFinding(
                    "edge.port_unknown",
                    f"Operation {operation_id!r} declares unknown {direction} port {port_id!r}.",
                    (f"$.plan.operations[{operation_id}].{direction}_refs",),
                    {"operation_id": operation_id, "port_id": port_id, "direction": direction},
                    "Use a named port published by the authoritative capability descriptor.",
                )
            )
            continue
        grouped[port_id].append(reference)
        metadata = dict(reference.get("metadata", {}) or {})
        if (
            str(reference.get("kind", "")) != port.artifact_kind
            or str(reference.get("schema_id", "")) not in port.schema_ids
            or any(metadata.get(key) is None for key in port.required_metadata)
        ):
            findings.append(
                EdgeCompilationFinding(
                    "edge.port_contract_mismatch",
                    f"Operation {operation_id!r} {direction} does not satisfy named port {port_id!r}.",
                    (f"$.plan.operations[{operation_id}].{direction}_refs",),
                    {
                        "operation_id": operation_id,
                        "port_id": port_id,
                        "direction": direction,
                        "reference": dict(reference),
                        "port": port.to_dict(),
                    },
                    "Preserve the port artifact kind, schema identity, and required metadata exactly.",
                )
            )
        if direction == "input" and available_sources is not None:
            source = available_sources.get(str(reference.get("ref_id", "")))
            if source is None:
                findings.append(
                    EdgeCompilationFinding(
                        "edge.input_source_not_direct",
                        f"Operation {operation_id!r} input is not a request input or direct dependency output.",
                        (f"$.plan.operations[{operation_id}].input_refs",),
                        {"operation_id": operation_id, "ref_id": reference.get("ref_id")},
                        "Bind the port to an exact request input or direct dependency output.",
                    )
                )
            elif any(
                reference.get(key) != source.get(key)
                for key in ("kind", "schema_id", "metadata")
            ):
                findings.append(
                    EdgeCompilationFinding(
                        "edge.input_source_identity_mismatch",
                        f"Operation {operation_id!r} input changes its source schema or metadata identity.",
                        (f"$.plan.operations[{operation_id}].input_refs",),
                        {
                            "operation_id": operation_id,
                            "ref_id": reference.get("ref_id"),
                            "declared": dict(reference),
                            "source": dict(source),
                        },
                        "Copy kind, schema_id, and metadata from the authoritative source unchanged.",
                    )
                )
        elif direction == "output":
            expected_metadata = {
                "producer_operation_id": operation_id,
                "semantic_role": port.semantic_role,
            }
            if metadata != expected_metadata:
                findings.append(
                    EdgeCompilationFinding(
                        "edge.output_metadata_mismatch",
                        f"Operation {operation_id!r} output metadata does not bind its producer and semantic role.",
                        (f"$.plan.operations[{operation_id}].output_refs",),
                        {
                            "operation_id": operation_id,
                            "port_id": port_id,
                            "expected_metadata": expected_metadata,
                            "observed_metadata": metadata,
                        },
                        "Use the host-authored producer_operation_id and semantic_role metadata.",
                    )
                )

    for port in ports:
        count = len(grouped[port.port_id])
        valid = (
            count == 1
            if port.cardinality == "exactly_one"
            else count <= 1
            if port.cardinality == "optional"
            else count >= 1
        )
        if not valid:
            findings.append(
                EdgeCompilationFinding(
                    "edge.port_cardinality_mismatch",
                    f"Operation {operation_id!r} has {count} references for {direction} port {port.port_id!r}.",
                    (f"$.plan.operations[{operation_id}].{direction}_refs",),
                    {
                        "operation_id": operation_id,
                        "port_id": port.port_id,
                        "direction": direction,
                        "cardinality": port.cardinality,
                        "count": count,
                    },
                    "Match the cardinality published by the authoritative capability descriptor.",
                )
            )
    return findings


def _request_source(item: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(item["kind"])
    metadata = deepcopy(dict(item.get("metadata", {}) or {}))
    content_sha256 = item.get("content_sha256")
    if content_sha256 is not None:
        metadata["content_sha256"] = str(content_sha256)
    else:
        metadata.pop("content_sha256", None)
    return {
        "ref_id": str(item["input_id"]),
        "kind": kind,
        "port_id": "request_input",
        "schema_id": str(item.get("schema_id") or f"oel.artifact.{kind}.v1"),
        "metadata": metadata,
    }


def _matches_port(source: Mapping[str, Any], port: CapabilityPort) -> bool:
    metadata = dict(source.get("metadata", {}) or {})
    return (
        str(source.get("kind")) == port.artifact_kind
        and str(source.get("schema_id")) in port.schema_ids
        and all(key in metadata and metadata[key] is not None for key in port.required_metadata)
    )


def _bind_input_ref(source: Mapping[str, Any], port_id: str) -> dict[str, Any]:
    return {
        "ref_id": str(source["ref_id"]),
        "kind": str(source["kind"]),
        "port_id": str(port_id),
        "schema_id": str(source["schema_id"]),
        "metadata": deepcopy(dict(source.get("metadata", {}) or {})),
    }


def _resource_bound_findings(
    operation_id: str,
    bounds: Mapping[str, Any],
    input_refs: Sequence[Mapping[str, Any]],
) -> list[EdgeCompilationFinding]:
    requirements: dict[str, float] = {}
    findings: list[EdgeCompilationFinding] = []
    metadata_to_bound = {
        "input_bytes": "max_input_bytes",
        "observation_count": "max_observations",
        "normal_point_count": "max_observations",
        "sample_count": "max_observations",
        "arc_duration_s": "max_arc_duration_s",
        "propagation_steps": "max_propagation_steps",
        "estimator_evaluations": "max_estimator_evaluations",
        "batch_evaluations": "max_batch_evaluations",
        "station_count": "max_stations",
    }
    for ref in input_refs:
        for metadata_name, bound_name in metadata_to_bound.items():
            value = dict(ref.get("metadata", {}) or {}).get(metadata_name)
            if value is not None:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError, OverflowError):
                    numeric_value = float("nan")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(numeric_value)
                    or numeric_value < 0.0
                ):
                    findings.append(
                        EdgeCompilationFinding(
                            "edge.resource_metadata_invalid",
                            f"Input metadata {metadata_name!r} for operation {operation_id!r} must be a finite non-negative number.",
                            (f"$.plan.operations[{operation_id}].input_refs",),
                            {
                                "operation_id": operation_id,
                                "metadata_name": metadata_name,
                                "value": value,
                            },
                            "Replace the resource metadata with the measured non-negative input bound.",
                        )
                    )
                    continue
                requirements[bound_name] = requirements.get(bound_name, 0.0) + numeric_value
    for bound_name, minimum in requirements.items():
        declared = bounds.get(bound_name)
        if declared is None:
            findings.append(
                EdgeCompilationFinding(
                    "edge.resource_bound_insufficient",
                    f"Operation {operation_id!r} does not bound {bound_name!r} for its selected inputs.",
                    (f"$.plan.operations[{operation_id}].bounds.{bound_name}",),
                    {
                        "operation_id": operation_id,
                        "bound": bound_name,
                        "declared": declared,
                        "required_minimum": minimum,
                    },
                    "Declare a per-operation maximum at least as large as the identity-bound input metadata.",
                )
            )
            continue
        try:
            numeric_declared = float(declared)
        except (TypeError, ValueError, OverflowError):
            numeric_declared = float("nan")
        if (
            isinstance(declared, bool)
            or not isinstance(declared, (int, float))
            or not math.isfinite(numeric_declared)
            or numeric_declared < 0.0
        ):
            findings.append(
                EdgeCompilationFinding(
                    "edge.resource_bound_invalid",
                    f"Operation {operation_id!r} declares an invalid {bound_name!r} resource bound.",
                    (f"$.plan.operations[{operation_id}].bounds.{bound_name}",),
                    {
                        "operation_id": operation_id,
                        "bound": bound_name,
                        "declared": declared,
                        "required_minimum": minimum,
                    },
                    "Declare a finite non-negative numeric bound before compiling typed edges.",
                )
            )
        elif numeric_declared < minimum:
            findings.append(
                EdgeCompilationFinding(
                    "edge.resource_bound_insufficient",
                    f"Operation {operation_id!r} does not bound {bound_name!r} for its selected inputs.",
                    (f"$.plan.operations[{operation_id}].bounds.{bound_name}",),
                    {
                        "operation_id": operation_id,
                        "bound": bound_name,
                        "declared": declared,
                        "required_minimum": minimum,
                    },
                    "Declare a per-operation maximum at least as large as the identity-bound input metadata.",
                )
            )
    return findings


def _graph_findings(operations: Sequence[Mapping[str, Any]]) -> list[EdgeCompilationFinding]:
    findings: list[EdgeCompilationFinding] = []
    operation_ids = [str(item.get("operation_id", "")) for item in operations]
    known = set(operation_ids)
    duplicate_ids = sorted({item for item in operation_ids if operation_ids.count(item) > 1})
    if duplicate_ids:
        findings.append(
            EdgeCompilationFinding(
                "edge.operation_id_duplicate",
                "Operation identifiers must be unique before typed edges can be compiled.",
                ("$.plan.operations",),
                {"operation_ids": duplicate_ids},
                "Assign one stable identifier to each semantic operation.",
            )
        )
    for item in operations:
        operation_id = str(item.get("operation_id", ""))
        unknown = sorted(str(value) for value in item.get("depends_on", []) if str(value) not in known)
        if unknown:
            findings.append(
                EdgeCompilationFinding(
                    "edge.dependency_unknown",
                    f"Operation {operation_id!r} names unknown dependencies.",
                    (f"$.plan.operations[{operation_id}].depends_on",),
                    {"operation_id": operation_id, "dependency_ids": unknown},
                    "Use only operation identifiers declared in this proposal.",
                )
            )
    if not findings and len(_topological_order(operations, allow_cycle=True)) != len(operations):
        findings.append(
            EdgeCompilationFinding(
                "edge.dependency_cycle",
                "The operation dependency graph contains a cycle.",
                ("$.plan.operations",),
                {},
                "Remove the cycle so OEL can establish a deterministic execution order.",
            )
        )
    return findings


def _topological_order(
    operations: Sequence[Mapping[str, Any]],
    *,
    allow_cycle: bool = False,
) -> list[str]:
    ordered_ids = [str(item["operation_id"]) for item in operations]
    dependencies = {
        str(item["operation_id"]): {str(value) for value in item["depends_on"]}
        for item in operations
    }
    completed: set[str] = set()
    result: list[str] = []
    while len(result) < len(operations):
        ready = [
            operation_id
            for operation_id in ordered_ids
            if operation_id not in completed and dependencies[operation_id].issubset(completed)
        ]
        if not ready:
            if allow_cycle:
                return result
            raise StudyEdgeCompilationError(
                (
                    EdgeCompilationFinding(
                        "edge.dependency_cycle",
                        "The operation dependency graph contains a cycle.",
                        ("$.plan.operations",),
                        {},
                        "Remove the cycle so OEL can establish a deterministic execution order.",
                    ),
                )
            )
        for operation_id in ready:
            result.append(operation_id)
            completed.add(operation_id)
    return result


__all__ = [
    "EDGE_COMPILATION_SCHEMA",
    "EdgeCompilationFinding",
    "StudyEdgeCompilationError",
    "compile_typed_operation_edges",
    "verify_typed_operation_edges",
]
