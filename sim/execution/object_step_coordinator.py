"""Deterministic serial object selection and compatibility metadata."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from sim.execution.object_workers import ObjectStepExecutor, SerialObjectStepExecutor
from sim.execution.worker_budget import worker_budget_value
from sim.pro_features import FEATURE_OBJECT_PARALLELISM, require_pro_feature

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine

_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD = 250.0

OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"

CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"

TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"

class ObjectStepCoordinator:
    """Select serial execution and retain the established public metadata."""

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        cpu_count: Callable[[], int | None] = os.cpu_count,
    ) -> None:
        self.engine = engine
        self.cpu_count = cpu_count

    def estimated_work_score(self, active_ids: list[str]) -> float:
        engine = self.engine
        dynamics = dict(getattr(engine.cfg.simulator, "dynamics", {}) or {})
        orbit = dict(dynamics.get("orbit", {}) or {})
        attitude = dict(dynamics.get("attitude", {}) or {})
        score = 1.0
        score += 0.5 * sum(bool(orbit.get(name, False)) for name in ("j2", "j3", "j4", "drag", "srp"))
        score += 0.75 * sum(bool(orbit.get(name, False)) for name in ("third_body_sun", "third_body_moon"))
        spherical = dict(orbit.get("spherical_harmonics", {}) or {})
        if bool(spherical.get("enabled", False)):
            degree = int(spherical.get("degree", 0) or 0)
            order = int(spherical.get("order", 0) or 0)
            score += 2.0 + 0.1 * float(max(degree, order))
        if bool(attitude.get("enabled", True)):
            attitude_substep_s = float(attitude.get("attitude_substep_s", engine.dt) or engine.dt)
            score += min(2.0, 0.5 * max(1.0, engine.dt / max(attitude_substep_s, 1e-9)))
        if active_ids:
            controlled = sum(
                engine.agents[aid].orbit_controller is not None
                or engine.agents[aid].attitude_controller is not None
                for aid in active_ids
            )
            knowledge = sum(engine.agents[aid].knowledge_base is not None for aid in active_ids)
            mission = sum(
                engine.agents[aid].mission_execution is not None
                or engine.agents[aid].mission_strategy is not None
                or bool(engine.agents[aid].mission_modules)
                for aid in active_ids
            )
            scale = float(len(active_ids))
            score += 1.0 * controlled / scale
            score += 0.5 * knowledge / scale
            score += 0.5 * mission / scale
        return float(score)

    def build_executor(self) -> ObjectStepExecutor:
        engine = self.engine
        execution_cfg = dict(getattr(engine.cfg.simulator, "execution", {}) or {})
        object_parallelism = dict(execution_cfg.get("object_parallelism", {}) or {})
        policy = str(execution_cfg.get("policy", "configured") or "configured").strip().lower()
        enabled = bool(object_parallelism.get("enabled", False))
        backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
        if policy != "serial" and enabled and backend not in {"serial", "process_pool", "thread_pool"}:
            raise ValueError(f"Unsupported object step executor backend: {backend!r}.")
        if policy in {"auto", "parallel"} or (
            policy != "serial" and enabled and backend in {"process_pool", "thread_pool"}
        ):
            require_pro_feature(FEATURE_OBJECT_PARALLELISM)
        active_ids = [aid for aid, agent in engine.agents.items() if agent.active]
        active_objects = len(active_ids)
        min_objects = int(object_parallelism.get("min_objects", 3) or 3)
        hierarchical_object_budget: int | None = None
        raw_hierarchical_budget = worker_budget_value(OBJECT_WORKER_BUDGET_ENV)
        if raw_hierarchical_budget not in (None, ""):
            hierarchical_object_budget = max(int(raw_hierarchical_budget), 0)
        incompatibilities: list[str] = []
        if any(
            agent.deploy_source in {"rocket_deployment", "rocket_insertion"}
            for agent in engine.agents.values()
        ):
            incompatibilities.append("dynamic object deployment is configured")
        if engine.general_propagation:
            incompatibilities.append("mixed OGP/general-propagation objects are configured")
        if any(agent.bridge is not None for agent in engine.agents.values()):
            incompatibilities.append("an external object bridge is configured")
        work_score = self.estimated_work_score(active_ids)
        planned_steps = max(
            1,
            int(
                np.ceil(
                    float(getattr(engine.cfg.simulator, "duration_s", 0.0))
                    / max(float(engine.dt), 1.0e-12)
                )
            ),
        )
        run_work_score = float(work_score) * float(planned_steps)
        reason = "simulator.execution.policy=serial" if policy == "serial" else "object parallelism is disabled"
        engine.object_execution_plan = {
            "policy": policy,
            "requested_backend": backend,
            "selected_backend": "serial",
            "selected_workers": 1,
            "active_objects": active_objects,
            "min_objects": min_objects,
            "estimated_work_score": float(work_score),
            "planned_steps": int(planned_steps),
            "estimated_run_work_score": float(run_work_score),
            "eligible": not incompatibilities,
            "reason": reason,
            "allocation": self.object_worker_allocation(active_ids, 1),
            "hierarchical_budget": {
                "object_workers_per_run": hierarchical_object_budget,
                "campaign_workers": (
                    None
                    if worker_budget_value(CAMPAIGN_WORKER_COUNT_ENV) in (None, "")
                    else int(worker_budget_value(CAMPAIGN_WORKER_COUNT_ENV))
                ),
                "total_process_budget": (
                    None
                    if worker_budget_value(TOTAL_PROCESS_BUDGET_ENV) in (None, "")
                    else int(worker_budget_value(TOTAL_PROCESS_BUDGET_ENV))
                ),
            },
        }
        return SerialObjectStepExecutor(engine)

    @staticmethod
    def object_worker_allocation(object_ids: list[str], workers: int) -> dict[str, list[str]]:
        if int(workers) > 1:
            require_pro_feature(FEATURE_OBJECT_PARALLELISM)
        return {"worker_1": [str(object_id) for object_id in object_ids]}
