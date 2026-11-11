# OEL Workflow Evidence Contract

This contract defines a generic, domain-neutral sidecar for completed OEL
workflows. It is an OEL interface for humans, CI, notebooks, reports, and
optional integrations. It is not an agent or model contract.

Domain artifacts remain authoritative. Each workflow keeps its validation
packet and provenance manifest. The generic `oel_workflow_evidence.json`
records how to locate and interpret those artifacts without duplicating
domain results.

Required sections are workflow identity, status and disposition, inputs,
quality gates, warnings, failures, artifact references with hashes, provenance,
data markings, non-claims, and a compact domain summary. Required artifact
existence is checked when the sidecar is built.

Normative schema:

- `docs/contracts/schemas/oel-workflow-evidence-v1.schema.json`

Dependency rule: optional integrations may consume this contract. No module
under `sim/` may import or require an optional analyst or model integration.
