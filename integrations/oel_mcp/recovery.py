"""Payload-free diagnostic guidance; never grants authority or changes inputs."""

from __future__ import annotations

from typing import TypeVar

from sim.security import ConfigPathSecurityError

E = TypeVar("E", bound=Exception)

RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "actor": {"enum": ["agent", "operator"]},
        "action": {"type": "string"},
    },
    "required": ["code", "actor", "action"],
    "additionalProperties": False,
}

# Messages are fixed public-safe guidance, never exception or argument payloads.
GUIDANCE = {
    "tool.arguments_invalid": ("agent", "Read the advertised input schema and repair arguments without changing study intent."),
    "tool.unavailable": ("agent", "Read active capability and workflow discovery; do not assume another deployment is authorized."),
    "handling.review_required": ("operator", "Establish authoritative handling metadata for this input; do not relabel data to bypass policy."),
    "approval.required": ("operator", "Supply an applicable operator-configured approval reference with the required scope; do not invent one."),
    "trust.required": ("operator", "Review source and paths and provide an applicable configured trust approval before importing code."),
    "entitlement.required": ("operator", "Verify the required local Pro entitlement before retrying this operation."),
    "path.not_authorized": ("operator", "Select an authorized path or have the operator configure appropriate roots; do not bypass containment."),
    "input.not_found": ("agent", "Check the selected input exists within authorized roots before retrying."),
    "output.already_exists": ("agent", "Inspect existing evidence and choose a new output target; do not overwrite or blindly repeat execution."),
    "scenario.policy_blocked": ("agent", "Review the reported settings against MCP policy and required evidence; edit and revalidate only if study intent is preserved."),
    "validation.stale": ("agent", "Validate the exact config, output and resource profile again; use its new validation ID with applicable approvals."),
    "input.invalid": ("agent", "Inspect the input contract and reported validation issue, preserve material study choices, then validate again."),
    "operation.conflict": ("operator", "Review the recorded operation identity and reconcile it before retrying; do not select a new authority."),
    "operation.unknown_outcome": ("operator", "Inspect and reconcile the reserved operation before retrying; do not rerun it under a new authority."),
}


def annotate_error(exc: E, code: str) -> E:
    if code not in GUIDANCE:
        raise ValueError("Unknown MCP recovery code.")
    exc.oel_recovery_code = code  # type: ignore[attr-defined]
    return exc


def recovery_details(exc: Exception) -> dict[str, str] | None:
    code = getattr(exc, "oel_recovery_code", None)
    if code not in GUIDANCE:
        code = (
            "path.not_authorized" if isinstance(exc, ConfigPathSecurityError) else
            "input.not_found" if isinstance(exc, FileNotFoundError) else
            "output.already_exists" if isinstance(exc, FileExistsError) else
            "input.invalid" if isinstance(exc, ValueError) else None
        )
    if code is None:
        return None
    actor, action = GUIDANCE[code]
    return {"code": code, "actor": actor, "action": action}


def reconciliation_error_data(exc: Exception) -> dict[str, object] | None:
    """Return stable, payload-free guidance for outcomes requiring reconciliation."""

    if not getattr(exc, "oel_reconciliation_required", False):
        return None
    data: dict[str, object] = {
        "reconciliation_required": True,
        "retryable": False,
    }
    recovery = recovery_details(exc)
    if recovery is not None:
        data["recovery"] = recovery
    return data
