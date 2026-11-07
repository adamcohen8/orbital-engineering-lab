from __future__ import annotations

import pytest

from integrations.oel_mcp.approval_grants import ApprovalGrantStore
from integrations.oel_mcp.execution import ExecutionApprovalPolicy


def test_unknown_explicit_write_grant_does_not_fall_back_to_standing_id() -> None:
    policy = ExecutionApprovalPolicy(
        write_approval_ids=frozenset({"standing-write"}),
        grant_store=ApprovalGrantStore(),
    )

    with pytest.raises(PermissionError):
        policy.authorize(
            {
                "approval_id": "standing-write",
                "grant_id": "missing-grant",
                "scope": "write",
            },
            executes=False,
            tool_id="tool.test.v1",
            arguments={},
        )


def test_unknown_explicit_trust_grant_does_not_fall_back_to_standing_id() -> None:
    policy = ExecutionApprovalPolicy(
        trust_approval_ids=frozenset({"standing-trust"}),
        grant_store=ApprovalGrantStore(),
    )

    with pytest.raises(PermissionError):
        policy.authorize_trust(
            {
                "approval_id": "standing-trust",
                "grant_id": "missing-grant",
                "scope": "trust",
            },
            tool_id="tool.test.v1",
            arguments={},
        )
