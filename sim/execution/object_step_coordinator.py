"""Object-step backend planning and executor selection."""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from sim.execution.object_workers import (
    ObjectStepExecutor,
    OGPBatchStepExecutor,
    ProcessPoolObjectStepExecutor,
    SerialObjectStepExecutor,
    ThreadPoolHistoryStepExecutor,
)
from sim.execution.worker_budget import worker_budget_value
from sim.pro_features import FEATURE_OBJECT_PARALLELISM, require_pro_feature
from sim.resource_limits import resource_profile

if TYPE_CHECKING:
    from sim.single_run import _SingleRunEngine

_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD = 250.0
OBJECT_WORKER_BUDGET_ENV = "OEL_OBJECT_WORKER_BUDGET"
CAMPAIGN_WORKER_COUNT_ENV = "OEL_CAMPAIGN_WORKER_COUNT"
TOTAL_PROCESS_BUDGET_ENV = "OEL_TOTAL_PROCESS_BUDGET"


class ObjectStepCoordinator:
    """Plan object backends while leaving lifecycle state on the engine."""

    def __init__(
        self,
        engine: _SingleRunEngine,
        *,
        cpu_count: Callable[[], int | None] = os.cpu_count,
    ) -> None:
        self.engine = engine
        self.cpu_count = cpu_count

    def build_executor(self) -> ObjectStepExecutor:
        engine = self.engine
        execution_cfg = dict(getattr(engine.cfg.simulator, "execution", {}) or {})
        object_parallelism = dict(execution_cfg.get("object_parallelism", {}) or {})
        policy = str(execution_cfg.get("policy", "configured") or "configured").strip().lower()
        enabled = bool(object_parallelism.get("enabled", False))
        backend = str(object_parallelism.get("backend", "serial") or "serial").strip().lower()
        thread_scope = str(object_parallelism.get("thread_scope", "auto") or "auto").strip().lower()
        profile = resource_profile(getattr(engine.cfg.simulator, "resource_profile", None))
        active_ids = [aid for aid, agent in engine.agents.items() if agent.active]
        active_objects = len(active_ids)
        min_objects = int(object_parallelism.get("min_objects", 3) or 3)
        workers = int(object_parallelism.get("workers", 0) or 0)
        if workers <= 0:
            reserve_workers = int(object_parallelism.get("reserve_workers", 1) or 0)
            workers = max(1, int(self.cpu_count() or 1) - max(0, reserve_workers))
        max_workers = int(object_parallelism.get("max_workers", 0) or 0)
        if max_workers > 0:
            workers = min(workers, max_workers)
        workers = max(1, min(workers, active_objects))
        if profile.max_parallel_workers is not None:
            workers = max(1, min(workers, int(profile.max_parallel_workers)))
        hierarchical_object_budget: int | None = None
        raw_hierarchical_budget = worker_budget_value(OBJECT_WORKER_BUDGET_ENV)
        if raw_hierarchical_budget not in (None, ""):
            hierarchical_object_budget = max(int(raw_hierarchical_budget), 0)
            workers = min(workers, hierarchical_object_budget) if hierarchical_object_budget > 0 else 1

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
        native_history_available = False
        if backend == "thread_pool":
            try:
                native = import_module("oel_rust_orbit")
                native_history_available = bool(getattr(native, "PASSIVE_HISTORY_GIL_RELEASED", False))
            except ImportError:
                pass
            cache = engine.satellite_stepper.passive_history
            unsupported = [aid for aid in active_ids if cache.thread_definition(
                engine.agents[aid], engine.agents[aid].truth,
            ) is None]
            native_history_available = native_history_available and not unsupported
            if thread_scope == "native_history" and not native_history_available:
                incompatibilities.append(
                    "thread_pool native_history requires passive trajectory_only ECI Rust RK4 two-body/J2 "
                    "objects and a GIL-releasing sampled-history binding"
                    + (": " + ", ".join(unsupported) if unsupported else "")
                )
            elif thread_scope == "object_step" or not native_history_available:
                from sim.execution.thread_contract import thread_worker_incompatibilities

                incompatibilities.extend(thread_worker_incompatibilities(engine, active_ids))
        if (backend == "thread_pool" or policy in {"auto", "parallel"}) and not incompatibilities:
            try:
                for aid in active_ids:
                    pickle.dumps(engine.agents[aid], protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                incompatibilities.append(
                    f"object {aid!r} runtime state is not process-serializable: {exc}"
                )

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
        selected_backend = "serial"
        reason = "object parallelism is disabled"

        if policy == "serial":
            reason = "simulator.execution.policy=serial"
        elif policy == "parallel":
            failures = list(incompatibilities)
            if bool(profile.force_serial):
                failures.append(f"resource profile {profile.name!r} requires serial execution")
            if active_objects < max(1, min_objects):
                failures.append(
                    f"active object count {active_objects} is below min_objects={min_objects}"
                )
            if workers <= 1:
                failures.append("fewer than two object workers are available")
            if failures:
                raise RuntimeError(
                    "Forced object-parallel execution is unavailable: " + "; ".join(failures)
                )
            selected_backend = "thread_pool" if backend == "thread_pool" else "process_pool"
            reason = "parallel policy explicitly requires supported object workers"
        elif policy == "auto":
            if bool(profile.force_serial):
                reason = f"resource profile {profile.name!r} requires serial execution"
            elif incompatibilities:
                reason = "auto planner selected serial: " + "; ".join(incompatibilities)
            elif active_objects < max(1, min_objects):
                reason = (
                    f"auto planner selected serial: active object count {active_objects} "
                    f"is below min_objects={min_objects}"
                )
            elif workers <= 1:
                reason = "auto planner selected serial: fewer than two object workers are available"
            elif work_score < 2.5:
                reason = (
                    "auto planner selected serial: estimated object-step work score "
                    f"{work_score:.2f} is below the 2.50 crossover threshold"
                )
            elif run_work_score < _AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD:
                reason = (
                    "auto planner selected serial: estimated run work score "
                    f"{run_work_score:.2f} is below the "
                    f"{_AUTO_OBJECT_PARALLEL_RUN_WORK_THRESHOLD:.2f} startup-amortization threshold"
                )
            else:
                selected_backend = "process_pool"
                reason = (
                    f"auto planner selected {workers} workers for {active_objects} active objects; "
                    f"estimated work score={work_score:.2f}"
                )
        elif enabled and backend in {"process_pool", "thread_pool"} and not bool(profile.force_serial):
            if incompatibilities:
                reason = "legacy configured object parallelism selected serial: " + "; ".join(
                    incompatibilities
                )
            elif active_objects >= max(1, min_objects) and workers > 1:
                selected_backend = backend
                reason = "legacy configured object parallelism is enabled"
            else:
                reason = "legacy configured object parallelism did not meet worker/object thresholds"
        elif enabled and backend not in {"serial", "process_pool", "thread_pool"}:
            raise ValueError(f"Unsupported object step executor backend: {backend!r}.")

        allocation = self.object_worker_allocation(
            active_ids,
            workers if selected_backend != "serial" else 1,
        )
        engine.object_execution_plan = {
            "policy": policy,
            "requested_backend": backend,
            "selected_backend": selected_backend,
            "selected_workers": workers if selected_backend != "serial" else 1,
            "active_objects": active_objects,
            "min_objects": min_objects,
            "estimated_work_score": float(work_score),
            "planned_steps": int(planned_steps),
            "estimated_run_work_score": float(run_work_score),
            "eligible": not incompatibilities,
            "reason": reason,
            "allocation": allocation,
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
        ogp_executor = self._build_ogp_executor(execution_cfg, policy)
        if ogp_executor is not None:
            return ogp_executor
        if selected_backend == "process_pool":
            require_pro_feature(FEATURE_OBJECT_PARALLELISM)
            return ProcessPoolObjectStepExecutor(engine, max_workers=workers)
        if selected_backend == "thread_pool":
            require_pro_feature(FEATURE_OBJECT_PARALLELISM)
            engine.object_execution_plan["thread_scope"] = thread_scope
            if native_history_available and thread_scope != "object_step":
                engine.object_execution_plan["parallel_scope"] = "native_history_preparation"
                return ThreadPoolHistoryStepExecutor(engine, max_workers=workers)
            from sim.execution.thread_workers import ThreadPoolObjectStepExecutor

            engine.object_execution_plan["parallel_scope"] = "object_step"
            return ThreadPoolObjectStepExecutor(engine, max_workers=workers)
        return SerialObjectStepExecutor(engine)

    def _build_ogp_executor(
        self, execution_cfg: dict, policy: str,
    ) -> ObjectStepExecutor | None:
        """Shared public OGP selection without enabling ONP/campaign workers."""
        engine = self.engine
        profile = resource_profile(getattr(engine.cfg.simulator, "resource_profile", None))
        ogp_parallelism = dict(execution_cfg.get("ogp_parallelism", {}) or {})
        if ogp_parallelism.get("enabled", False):
            from sim.rust_ogp_backend import resolve_ogp_batch_workers

            allocated = resolve_ogp_batch_workers(ogp_parallelism.get("workers", 0))
            if profile.max_parallel_workers is not None:
                allocated = min(allocated, max(1, int(profile.max_parallel_workers)))
            if profile.force_serial or policy == "serial":
                allocated = 1
            engine.object_execution_plan.update(
                requested_backend="native_threads", parallel_scope="ogp_history_preparation",
                eligible=True, workers_allocated=allocated,
                reason="OGP history preparation has not run yet; owner stepping is serial",
            )
            return OGPBatchStepExecutor(engine, max_workers=allocated)
        return None

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

    @staticmethod
    def object_worker_allocation(object_ids: list[str], workers: int) -> dict[str, list[str]]:
        count = max(1, int(workers))
        allocation = {f"worker_{index + 1}": [] for index in range(count)}
        for index, object_id in enumerate(object_ids):
            allocation[f"worker_{index % count + 1}"].append(str(object_id))
        return allocation
