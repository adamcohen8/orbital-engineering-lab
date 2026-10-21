"""Execution-local worker budgets for threads and process environment fallback.

Process launchers retain their existing inherited environment contract. Thread
launchers install immutable values in a context rather than changing the
environment shared by every run in the process.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"
CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"
TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"


@dataclass(frozen=True)
class WorkerBudget:
    object_workers_per_run: int
    campaign_workers: int
    total_process_budget: int


@dataclass(frozen=True)
class ThreadCampaignRuntimeScope:
    """Execution-local marker for a complete scenario in a campaign thread."""

    label: str


_WORKER_BUDGET: ContextVar[WorkerBudget | None] = ContextVar("oel_worker_budget", default=None)
_THREAD_CAMPAIGN_SCOPE: ContextVar[ThreadCampaignRuntimeScope | None] = ContextVar(
    "oel_thread_campaign_runtime_scope",
    default=None,
)


def current_worker_budget() -> WorkerBudget | None:
    """Return only an explicitly installed execution-local budget."""
    return _WORKER_BUDGET.get()


def current_thread_campaign_runtime_scope() -> ThreadCampaignRuntimeScope | None:
    """Return the current thread campaign marker, if one is installed."""

    return _THREAD_CAMPAIGN_SCOPE.get()


def worker_budget_value(name: str) -> str | None:
    """Resolve a scoped value before the legacy process environment."""
    budget = _WORKER_BUDGET.get()
    if budget is not None:
        fields = {
            OBJECT_WORKER_BUDGET_ENV: budget.object_workers_per_run,
            CAMPAIGN_WORKER_COUNT_ENV: budget.campaign_workers,
            TOTAL_PROCESS_BUDGET_ENV: budget.total_process_budget,
        }
        if name in fields:
            return str(fields[name])
    return os.environ.get(name)


@contextmanager
def worker_budget_context(
    *, object_workers_per_run: int, campaign_workers: int, total_process_budget: int,
) -> Iterator[WorkerBudget]:
    """Install immutable per-run allocation and restore the calling context."""
    values = (int(object_workers_per_run), int(campaign_workers), int(total_process_budget))
    if values[0] < 0 or values[1] < 1 or values[2] < 1:
        raise ValueError("Worker budgets require nonnegative object slots and positive campaign/total slots.")
    budget = WorkerBudget(*values)
    token = _WORKER_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _WORKER_BUDGET.reset(token)


@contextmanager
def thread_campaign_runtime_scope(label: str = "thread campaign scenario") -> Iterator[None]:
    """Install an execution-local complete-scenario thread marker."""

    token = _THREAD_CAMPAIGN_SCOPE.set(
        ThreadCampaignRuntimeScope(label=str(label or "thread campaign scenario"))
    )
    try:
        yield
    finally:
        _THREAD_CAMPAIGN_SCOPE.reset(token)


__all__ = [
    "ThreadCampaignRuntimeScope",
    "WorkerBudget",
    "current_thread_campaign_runtime_scope",
    "current_worker_budget",
    "thread_campaign_runtime_scope",
    "worker_budget_context",
    "worker_budget_value",
]
