from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from sim.execution.worker_budget import worker_budget_value
from sim.resource_limits import ResourceProfile, resource_profile

OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"
CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"
TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"


@dataclass(frozen=True)
class HierarchicalExecutionPlan:
    total_process_budget: int
    campaign_workers: int
    object_workers_per_run: int
    task_count: int
    forced_inner_parallel: bool
    inner_parallel_eligible: bool
    reason: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def restrictive_profile_for_task_roots(
    task_roots: Iterable[dict[str, Any]],
    *,
    fallback: str = "config",
) -> ResourceProfile:
    profiles = []
    for root in task_roots:
        simulator = dict(dict(root or {}).get("simulator", {}) or {})
        profiles.append(resource_profile(simulator.get("resource_profile", fallback)))
    if not profiles:
        return resource_profile(fallback)
    forced = next((profile for profile in profiles if profile.force_serial), None)
    if forced is not None:
        return forced
    capped = [profile for profile in profiles if profile.max_parallel_workers is not None]
    if capped:
        return min(capped, key=lambda profile: int(profile.max_parallel_workers or 1))
    return resource_profile(fallback)


def _object_parallel_request(root: dict[str, Any]) -> tuple[str, int, bool]:
    simulator = dict(root.get("simulator", {}) or {})
    execution = dict(simulator.get("execution", {}) or {})
    objects = dict(execution.get("object_parallelism", {}) or {})
    policy = str(execution.get("policy", "configured") or "configured").strip().lower()
    enabled = objects.get("enabled", False)
    if isinstance(enabled, str) and enabled.strip().lower() == "auto":
        policy = "auto"
    backend = str(objects.get("backend", "serial") or "serial").strip().lower()
    capable = policy in {"auto", "parallel"} or (bool(enabled) and backend in {"process_pool", "thread_pool"})
    if not capable:
        return policy, 0, False
    requested = int(objects.get("workers", 0) or objects.get("max_workers", 0) or 0)
    return policy, max(requested, 0), True


def plan_hierarchical_execution(
    *,
    task_roots: Iterable[dict[str, Any]],
    task_count: int,
    requested_campaign_workers: int,
    profile: ResourceProfile,
    reserve_processes: int = 1,
) -> HierarchicalExecutionPlan:
    tasks = max(int(task_count), 1)
    cpu_budget = max(1, int(os.cpu_count() or 1) - max(int(reserve_processes), 0))
    total_budget = cpu_budget
    if profile.force_serial:
        total_budget = 1
    elif profile.max_parallel_workers is not None:
        total_budget = max(1, min(total_budget, int(profile.max_parallel_workers)))
    inherited_object_budget = worker_budget_value(OBJECT_WORKER_BUDGET_ENV)
    if inherited_object_budget not in (None, ""):
        # A campaign launched inside an existing campaign branch owns its
        # current orchestrator slot plus only the child slots assigned to it.
        branch_budget = 1 + max(int(inherited_object_budget), 0)
        total_budget = max(1, min(total_budget, branch_budget))

    # Only compact policy tuples are retained.  Campaign roots may be numerous
    # and are already owned by the caller; copying the whole population here
    # can make resource planning itself the source of memory pressure.
    requests = [_object_parallel_request(dict(root or {})) for root in task_roots]
    forced = any(policy == "parallel" for policy, _requested, _capable in requests)
    inner_capable = any(capable for _policy, _requested, capable in requests)
    requested_outer = int(requested_campaign_workers)
    if requested_outer <= 0:
        requested_outer = total_budget
    campaign_workers = max(1, min(requested_outer, tasks, total_budget))

    if forced:
        max_campaigns_with_inner_workers = total_budget // 3
        if max_campaigns_with_inner_workers < 1:
            raise RuntimeError(
                "Hierarchical execution cannot satisfy forced inner parallelism: "
                f"total process budget {total_budget} cannot provide one campaign worker plus two object workers."
            )
        campaign_workers = min(campaign_workers, max_campaigns_with_inner_workers)

    remaining = max(total_budget - campaign_workers, 0)
    object_workers = remaining // campaign_workers if inner_capable else 0
    requested_inner_caps = [requested for _policy, requested, _capable in requests if requested > 0]
    if requested_inner_caps and object_workers > 0:
        object_workers = min(object_workers, max(requested_inner_caps))
    if object_workers < 2:
        object_workers = 0
    if forced and object_workers < 2:
        raise RuntimeError(
            "Hierarchical execution cannot satisfy forced inner parallelism after campaign allocation: "
            f"budget={total_budget}, campaign_workers={campaign_workers}."
        )

    if profile.force_serial:
        reason = f"resource profile {profile.name!r} requires serial campaign and object execution"
    elif object_workers > 0:
        reason = (
            f"allocated {campaign_workers} campaign workers and up to {object_workers} object workers per run "
            f"within total process budget {total_budget}"
        )
    elif inner_capable:
        reason = (
            f"allocated all {total_budget} available process slots to independent campaign runs; "
            "nested object execution is capped to serial"
        )
    else:
        reason = f"allocated {campaign_workers} campaign workers; child scenarios do not request object parallelism"
    return HierarchicalExecutionPlan(
        total_process_budget=total_budget,
        campaign_workers=campaign_workers,
        object_workers_per_run=object_workers,
        task_count=tasks,
        forced_inner_parallel=forced,
        inner_parallel_eligible=inner_capable,
        reason=reason,
    )


def apply_hierarchical_worker_env(plan: HierarchicalExecutionPlan) -> dict[str, str | None]:
    values = {
        OBJECT_WORKER_BUDGET_ENV: str(int(plan.object_workers_per_run)),
        CAMPAIGN_WORKER_COUNT_ENV: str(int(plan.campaign_workers)),
        TOTAL_PROCESS_BUDGET_ENV: str(int(plan.total_process_budget)),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    return previous


def restore_hierarchical_worker_env(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
