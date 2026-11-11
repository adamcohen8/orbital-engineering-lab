"""Local OEL analytical MCP handlers."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

import yaml

from integrations.oel_mcp.base_handlers import require_file_size
from integrations.oel_mcp.contracts import (
    MAX_MANIFEST_BYTES,
    MAX_REVIEW_STORE_BYTES,
    ToolContract,
)
from integrations.oel_mcp.execution import (
    MCPExecutionCancelled,
    cancellation_callback,
    complete_manifest,
    manifest_base,
    write_execution_manifest,
)
from integrations.oel_mcp.secure_writes import (
    OutputWriter,
    SecureWriteRoot,
    bind_output_writer,
)
from sim.estimation import solve_dynamics_orbit_determination
from sim.observations import load_observation_packet
from sim.pro_trajectory_optimization import (
    TrajectoryOptimizationProblem,
    solve_trajectory_optimization,
)
from sim.pro_trajectory_optimization import (
    replay_evidence as replay_trajectory_optimization_evidence,
)
from sim.pro_trajectory_optimization import (
    write_evidence as write_trajectory_optimization_evidence,
)
from sim.scale.planner import load_scale_planner_config, resolve_selected_object_ids
from sim.scale.recompute import run_recompute_plan
from sim.scale.store import summarize_store
from sim.scale.workflow import run_refresh_to_onp_workflow

M5_SCALE_MAX_SELECTED_OBJECTS = 1000
M5_RESOURCE_PROFILE = "laptop-safe"


class LocalAnalyticsMCPHandlers:
    """Execute bounded local analysis using the engine and MCP path policy."""

    _execute_scale_refresh = staticmethod(run_refresh_to_onp_workflow)

    def _call_local_analytics(self, contract, arguments, *, cancel_event=None, progress=None):
        operation = contract.tool_id.removeprefix("oel.").removeprefix("pro.").removesuffix(".v1")
        methods = {
            "inspect_scale_catalog": (self._inspect_scale_catalog, False),
            "plan_scale_recompute": (self._plan_scale_recompute, True),
            "inspect_scale_recompute_plan": (self._inspect_scale_recompute_plan, False),
            "validate_scale_refresh": (self._validate_scale_refresh, False),
            "run_scale_refresh_to_onp": (self._run_scale_refresh, True),
            "inspect_scale_refresh": (self._inspect_scale_refresh, False),
            "run_dynamics_od": (self._run_dynamics_od, True),
            "validate_trajectory_optimization": (self._validate_trajectory_optimization, False),
            "run_trajectory_optimization": (self._run_trajectory_optimization, True),
            "replay_trajectory_optimization": (self._replay_trajectory_optimization, True),
        }
        method = methods.get(operation)
        if method is None:
            return None
        run, supports_progress = method
        if supports_progress:
            return run(contract, arguments, cancel_event=cancel_event, progress=progress)
        return run(contract, arguments)

    def _validate_trajectory_optimization(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["problem_path"], kind="file")
            require_file_size(source, maximum=5_000_000)
            problem = TrajectoryOptimizationProblem.from_json(source)
            normalized = problem.to_dict()
            resource_plan = _trajectory_resource_plan(problem)
            return {
                "status": "validated",
                "problem": normalized,
                "problem_sha256": _semantic_sha256(normalized),
                "resource_plan": resource_plan,
                "execution_ready": resource_plan["action"] == "proceed",
                "execution_authorized": False,
                "hidden_truth_visible": False,
                **self._m5_policy_projection(contract, arguments),
                "non_claims": [
                    "Validation does not solve the optimization problem or establish feasibility.",
                    "A solver success flag is insufficient without authoritative repropagation and evidence replay.",
                ],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda result: {"complete": bool(result["execution_ready"]), "empty": False, "truncated": False},
        )


    def _run_trajectory_optimization(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["problem_path"], kind="file")
            require_file_size(source, maximum=5_000_000)
            problem = TrajectoryOptimizationProblem.from_json(source)
            resource_plan = _trajectory_resource_plan(problem)
            if resource_plan["action"] != "proceed":
                raise ValueError("Trajectory optimization exceeds the bounded laptop-safe resource contract.")
            self._enforce_approval_resource_ceiling(resource_plan)
            output = self.path_policy.resolve_write(arguments["output_dir"])
            check = cancellation_callback(cancel_event)
            check()
            if progress:
                progress(1, 3, "Validated the bounded trajectory-optimization problem.")
            raw = solve_trajectory_optimization(problem)
            check()
            if progress:
                progress(2, 3, "Completed optimization and authoritative ONP repropagation.")
            with SecureWriteRoot.open(
                output,
                approved_roots=self.path_policy.write_roots,
                create=True,
                require_empty=True,
                active_opid=self.active_operation_id,
            ) as writer:
                # The producer's manifest inventories relative filenames and
                # hashes only.  Its direct writer preserves the final logical
                # output path while protecting every physical write.
                manifest = write_trajectory_optimization_evidence(
                    raw,
                    output,
                    writer=writer,
                )
                # The content-bound producer returns its exact relative
                # inventory.  Do not rediscover artifacts through the visible
                # path after descriptor-bound writes; a concurrent rename of
                # that path must not change the response envelope.
                artifacts = [
                    str(output / str(entry["path"]))
                    for entry in manifest["files"]
                ] + [str(output / "trajectory_optimization_manifest.json")]
                if progress:
                    progress(3, 3, "Wrote the replayable trajectory-optimization evidence packet.")
                converged = bool(raw.get("converged", False)) and str(raw.get("status", "")) == "converged"
                summary = {
                    key: raw[key]
                    for key in (
                        "schema_version",
                        "status",
                        "converged",
                        "problem_sha256",
                        "objective",
                        "solver",
                        "solution",
                        "constraint_summary",
                        "robustness_summary",
                    )
                    if key in raw
                }
                return {
                    "status": "completed" if converged else "partial",
                    "solution_status": str(raw.get("status", "")),
                    "converged": converged,
                    "problem_sha256": str(raw.get("problem_sha256", "")),
                    "output_dir": str(output),
                    "summary": summary,
                    "manifest": dict(manifest),
                    "artifacts": artifacts,
                    "authoritative_replay_required": True,
                    "hidden_truth_visible": False,
                    **self._m5_policy_projection(contract, arguments),
                    "non_claims": [
                        "Local convergence is not a claim of global optimality.",
                        "Optimization evidence does not provide maneuver authority or operational qualification.",
                        "The retained packet must pass authoritative replay before its solution is cited.",
                    ],
                }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: str(result["status"]),
            evidence=lambda result: {"complete": result["status"] == "completed", "empty": False, "truncated": False},
        )


    def _replay_trajectory_optimization(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            entries = list(output.iterdir())
            if len(entries) > 16:
                raise ValueError("Trajectory-optimization evidence packet exceeds the file-count limit.")
            total_bytes = 0
            for entry in entries:
                resolved = self.path_policy.resolve_read_child(output, entry.name, kind="file")
                require_file_size(resolved, maximum=50_000_000)
                total_bytes += resolved.stat().st_size
            if total_bytes > 100_000_000:
                raise ValueError("Trajectory-optimization evidence packet exceeds the replay byte limit.")
            check = cancellation_callback(cancel_event)
            check()
            if progress:
                progress(1, 2, "Verified trajectory-optimization evidence inventory and digests.")
            raw = replay_trajectory_optimization_evidence(output)
            check()
            if progress:
                progress(2, 2, "Completed authoritative optimization and repropagation replay.")
            return {
                **dict(raw),
                "output_dir": str(output),
                "hidden_truth_visible": False,
                **self._m5_policy_projection(contract, arguments),
                "non_claims": [
                    "Replay establishes deterministic evidence identity, not global optimality.",
                    "Verified optimization evidence does not provide maneuver authority or operational qualification.",
                ],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )


    def _run_dynamics_od(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            packet_path = self.path_policy.resolve_read(arguments["observation_packet_path"], kind="file")
            require_file_size(packet_path, maximum=MAX_MANIFEST_BYTES)
            packet = load_observation_packet(packet_path)
            observations = list(packet.observations)
            observation_count = len(observations)
            if observation_count < 2 or observation_count > 2_000:
                raise ValueError("Dynamics OD requires 2 to 2,000 bounded observations.")
            if bool(arguments["detect_maneuvers"]) and observation_count > 500:
                raise ValueError("Maneuver-search OD is limited to 500 observations.")
            times = [float(row["time_s"]) for row in observations]
            if not all(math.isfinite(value) for value in times):
                raise ValueError("Dynamics OD observation epochs must be finite.")
            arc_duration_s = max(times) - min(times)
            if arc_duration_s < 0.0 or arc_duration_s > 604_800.0:
                raise ValueError("Dynamics OD observation arcs are limited to seven days.")
            requested_duration_s = float(arguments["fit_duration_s"]) + float(
                arguments["holdout_duration_s"]
            )
            if requested_duration_s > 604_800.0:
                raise ValueError("Dynamics OD fit plus holdout duration is limited to seven days.")
            resource_plan = {
                "resource_profile": "laptop-safe",
                "observation_count": observation_count,
                "max_observations": 2_000,
                "arc_duration_s": arc_duration_s,
                "max_arc_duration_s": 604_800.0,
                "max_nfev": int(arguments["max_nfev"]),
                "maneuver_search": bool(arguments["detect_maneuvers"]),
                "action": "proceed",
            }
            self._enforce_approval_resource_ceiling(resource_plan)
            output = self.path_policy.resolve_write(arguments["output_dir"])
            with SecureWriteRoot.open(
                output,
                approved_roots=self.path_policy.write_roots,
                create=True,
                require_empty=True,
                active_opid=self.active_operation_id,
            ) as writer:
                with bind_output_writer(writer):
                    manifest = manifest_base(
                        tool_id=contract.tool_id,
                        approval_id=str(arguments["approval"]["approval_id"]),
                    )
                    manifest.update({
                        "observation_packet_sha256": _sha256_file(packet_path),
                        "max_nfev": int(arguments["max_nfev"]),
                        "resource_plan": resource_plan,
                        "execution_authorized_outputs": False,
                        "data_label": str(
                            dict(arguments.get("handling", {}) or {}).get("data_label", "")
                        ),
                        "hidden_truth_included": False,
                    })
                    manifest_path = output / "mcp_execution_manifest.json"
                    write_execution_manifest(output, manifest, writer=writer)
                    check = cancellation_callback(cancel_event)
                    try:
                        check()
                        if progress:
                            progress(1, 2, "Loaded and bounded the OEL observation packet.")
                        raw = solve_dynamics_orbit_determination(
                            packet,
                            output_dir=output,
                            object_id=arguments.get("object_id"),
                            fit_duration_s=float(arguments["fit_duration_s"]),
                            holdout_duration_s=float(arguments["holdout_duration_s"]),
                            dt_s=float(arguments["dt_s"]),
                            dynamics_model=str(arguments["dynamics_model"]),
                            j2=bool(arguments["j2"]),
                            max_nfev=int(arguments["max_nfev"]),
                            detect_maneuvers=bool(arguments["detect_maneuvers"]),
                        )
                        check()
                        emission = dict(raw.get("state_estimate_product_emission", {}) or {})
                        artifacts = _dynamics_od_artifacts(output, raw, writer=writer)
                        status = "completed" if emission.get("status") == "emitted" else "partial"
                        complete_manifest(
                            manifest,
                            status=status,
                            artifacts=artifacts,
                            writer=writer,
                        )
                        write_execution_manifest(output, manifest, writer=writer)
                        if progress:
                            progress(2, 2, "Completed dynamics OD and typed product emission.")
                        return {
                            "status": status,
                            "method": str(raw.get("method", "")),
                            "quality_gates": dict(raw.get("quality_gates", {}) or {}),
                            "verdict": dict(raw.get("verdict", {}) or {}),
                            "fit_metrics": dict(raw.get("fit_metrics", {}) or {}),
                            "holdout_metrics": dict(raw.get("holdout_metrics", {}) or {}),
                            "maneuver_detection": dict(raw.get("maneuver_detection", {}) or {}),
                            "resource_plan": resource_plan,
                            "state_product_path": str(emission.get("product_path", "")),
                            "artifacts": artifacts,
                            "operation_manifest_path": str(manifest_path),
                            "execution_authorized_outputs": False,
                            "non_claims": [
                                "An OD fit is not operational orbit certification.",
                                "Materialized scenarios and state products require separate validation and execution approval.",
                            ],
                        }
                    except MCPExecutionCancelled:
                        artifacts = _dynamics_od_artifacts(output, {}, writer=writer)
                        complete_manifest(
                            manifest,
                            status="cancelled",
                            cancelled=True,
                            artifacts=artifacts,
                            writer=writer,
                            error_type="MCPExecutionCancelled",
                        )
                        write_execution_manifest(output, manifest, writer=writer)
                        return {
                            "status": "cancelled", "method": "dynamics_orbit_least_squares",
                            "quality_gates": {}, "verdict": {}, "fit_metrics": {}, "holdout_metrics": {},
                            "maneuver_detection": {}, "resource_plan": resource_plan,
                            "state_product_path": "", "artifacts": artifacts,
                            "operation_manifest_path": str(manifest_path),
                            "execution_authorized_outputs": False,
                            "non_claims": ["Cancelled OD evidence is incomplete."],
                        }
                    except Exception as exc:
                        complete_manifest(manifest, status="failed", writer=writer, error_type=type(exc).__name__)
                        write_execution_manifest(output, manifest, writer=writer)
                        raise

        return self._envelope(
            contract=contract, arguments=arguments, operation=operation,
            outcome_status=lambda result: result["status"] if result["status"] in {"completed", "failed"} else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed", "empty": False, "truncated": False
            },
        )


    def _inspect_scale_catalog(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            store = self.path_policy.resolve_read(arguments["store_path"], kind="file")
            require_file_size(store, maximum=MAX_REVIEW_STORE_BYTES)
            summary = summarize_store(store)
            return {
                "status": "completed",
                "schema_version": summary.schema_version,
                "store_snapshot_sha256": _scale_store_snapshot_sha256(summary, store),
                "catalog_object_count": summary.catalog_object_count,
                "current_object_state_count": summary.current_object_state_count,
                "conjunction_candidate_count": summary.conjunction_candidate_count,
                "open_invalidation_count": summary.open_invalidation_count,
                "invalidation_by_reason": dict(summary.invalidation_by_reason),
                "table_counts": dict(summary.table_counts),
                "hidden_truth_visible": False,
                **self._m5_policy_projection(contract, arguments),
                "non_claims": [
                    "Aggregate catalog health is not conjunction probability or operational decision authority.",
                    "Object identities and row payloads are not returned by this bounded inspection tool.",
                ],
            }

        return self._envelope(contract=contract, arguments=arguments, operation=operation)


    def _plan_scale_recompute(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            config = self.path_policy.resolve_read(arguments["config_path"], kind="file")
            require_file_size(config, maximum=MAX_MANIFEST_BYTES)
            output = self.path_policy.resolve_write(arguments["output_dir"])
            cfg = load_scale_planner_config(config, output_dir=output)
            if cfg.mode != "run":
                raise ValueError("M5 Scale recompute planning requires scale_catalog.mode: run.")
            store = self.path_policy.resolve_read(cfg.store_path, kind="file")
            require_file_size(store, maximum=MAX_REVIEW_STORE_BYTES)
            selected_object_ids = resolve_selected_object_ids(cfg)
            if len(selected_object_ids) > M5_SCALE_MAX_SELECTED_OBJECTS:
                raise ValueError("The Scale selection exceeds the M5 bounded planning limit.")
            resource_plan = {
                "resource_profile": M5_RESOURCE_PROFILE,
                "workers": 1,
                "selected_object_count": len(selected_object_ids),
                "max_selected_objects": M5_SCALE_MAX_SELECTED_OBJECTS,
                "action": "proceed",
            }
            self._enforce_approval_resource_ceiling(resource_plan)
            with SecureWriteRoot.open(
                output,
                approved_roots=self.path_policy.write_roots,
                create=True,
                require_empty=True,
                active_opid=self.active_operation_id,
            ) as writer:
                approval_id = str(arguments["approval"]["approval_id"])
                manifest = manifest_base(tool_id=contract.tool_id, approval_id=approval_id)
                manifest.update(
                    {
                        "output_dir": str(output),
                        "selected_object_count": len(selected_object_ids),
                        "config_sha256": _sha256_file(config),
                        "store_snapshot_sha256": _scale_store_snapshot_sha256(summarize_store(store), store),
                        "data_label": _data_label(arguments),
                        "hidden_truth_included": False,
                    }
                )
                manifest_path = write_execution_manifest(output, manifest, writer=writer)
                callback = cancellation_callback(cancel_event)
                try:
                    callback()
                    if progress:
                        progress(1, 2, "Validated the bounded Scale recompute request.")
                    raw = run_recompute_plan(config, output_dir=output, writer=writer)
                    callback()
                    artifacts = [
                        str(output / "scale_recompute_plan.json"),
                        str(output / "scale_recompute_plan.md"),
                    ]
                    complete_manifest(manifest, status="completed", artifacts=artifacts, writer=writer)
                    write_execution_manifest(output, manifest, writer=writer)
                    if progress:
                        progress(2, 2, "Completed the targeted Scale recompute plan.")
                    return {
                        "status": "completed",
                        "workflow": {
                            "name": dict(raw.get("workflow", {}) or {}).get("name"),
                            "mode": dict(raw.get("workflow", {}) or {}).get("mode"),
                        },
                        "selection": {
                            "selected_object_count": int(
                                dict(raw.get("selection", {}) or {}).get("selected_object_count", 0) or 0
                            )
                        },
                        "invalidation": {
                            "open_items": int(dict(raw.get("invalidation", {}) or {}).get("open_items", 0) or 0),
                            "by_reason": dict(dict(raw.get("invalidation", {}) or {}).get("by_reason", {}) or {}),
                        },
                        "work_counts": _scale_work_counts(raw),
                        "planned_jobs": _scale_job_projection(list(raw.get("planned_jobs", []) or [])),
                        "artifacts": artifacts,
                        "manifest_path": str(manifest_path),
                        "hidden_truth_visible": False,
                        **self._m5_policy_projection(contract, arguments),
                        "non_claims": list(raw.get("non_claims", []) or []),
                    }
                except MCPExecutionCancelled:
                    artifacts = [
                        str(output / "scale_recompute_plan.json"),
                        str(output / "scale_recompute_plan.md"),
                    ]
                    complete_manifest(
                        manifest,
                        status="cancelled",
                        cancelled=True,
                        artifacts=artifacts,
                        writer=writer,
                        error_type="MCPExecutionCancelled",
                    )
                    write_execution_manifest(output, manifest, writer=writer)
                    return {
                        "status": "cancelled",
                        "workflow": {"name": cfg.name, "mode": cfg.mode},
                        "selection": {"selected_object_count": len(selected_object_ids)},
                        "invalidation": {},
                        "work_counts": {},
                        "planned_jobs": [],
                        "artifacts": artifacts,
                        "manifest_path": str(manifest_path),
                        "hidden_truth_visible": False,
                        **self._m5_policy_projection(contract, arguments),
                        "non_claims": ["Cancelled Scale planning is incomplete and does not authorize recompute execution."],
                    }
                except Exception as exc:
                    complete_manifest(manifest, status="failed", writer=writer, error_type=type(exc).__name__)
                    write_execution_manifest(output, manifest, writer=writer)
                    raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": not bool(result["planned_jobs"]),
                "truncated": False,
            },
        )


    def _inspect_scale_recompute_plan(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            plan_path = self.path_policy.resolve_read_child(output, "scale_recompute_plan.json", kind="file")
            manifest_path = self.path_policy.resolve_read_child(
                output, "mcp_execution_manifest.json", kind="file", required=False
            )
            require_file_size(plan_path, maximum=MAX_MANIFEST_BYTES)
            plan = _load_json_object(plan_path)
            manifest = _load_optional_manifest(manifest_path)
            manifest["artifacts_complete"] = _manifest_artifacts_valid(manifest, output)
            status = str(manifest.get("status", plan.get("status", "partial")))
            if status == "completed" and not bool(manifest.get("artifacts_complete", False)):
                status = "partial"
            return {
                "status": status if status in {"completed", "partial", "failed", "cancelled"} else "partial",
                "workflow": {
                    "name": dict(plan.get("workflow", {}) or {}).get("name"),
                    "mode": dict(plan.get("workflow", {}) or {}).get("mode"),
                },
                "selection": {
                    "selected_object_count": int(
                        dict(plan.get("selection", {}) or {}).get("selected_object_count", 0) or 0
                    )
                },
                "invalidation": {
                    "open_items": int(dict(plan.get("invalidation", {}) or {}).get("open_items", 0) or 0),
                    "by_reason": dict(dict(plan.get("invalidation", {}) or {}).get("by_reason", {}) or {}),
                },
                "work_counts": _scale_work_counts(plan),
                "planned_jobs": _scale_job_projection(list(plan.get("planned_jobs", []) or [])),
                "manifest": _manifest_projection(manifest),
                "hidden_truth_visible": False,
                **self._m5_policy_projection(contract, arguments),
                "non_claims": [
                    "This inspection does not execute propagation, screening, refinement, or ONP.",
                    "Object identities and row payloads are not returned.",
                    "A recompute plan is not conjunction probability or operational decision authority.",
                ],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed" and bool(result["manifest"].get("artifacts_complete")),
                "empty": not bool(result["planned_jobs"]),
                "truncated": False,
            },
        )


    def _validate_scale_refresh(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            config, cfg, store, selected = self._bounded_scale_request(arguments)
            output = self.path_policy.resolve_write(arguments["output_dir"])
            resource_plan = {
                "resource_profile": M5_RESOURCE_PROFILE,
                "workers": 1,
                "selected_object_count": len(selected),
                "max_selected_objects": M5_SCALE_MAX_SELECTED_OBJECTS,
                "action": "proceed" if len(selected) <= M5_SCALE_MAX_SELECTED_OBJECTS else "refuse",
            }
            self._enforce_approval_resource_ceiling(resource_plan)
            validation_id = _scale_refresh_validation_id(
                config=config,
                store=store,
                selected_object_ids=selected,
                data_label=_data_label(arguments),
            )
            ready = resource_plan["action"] == "proceed"
            with SecureWriteRoot.open(
                output,
                approved_roots=self.path_policy.write_roots,
                create=True,
                require_empty=True,
                active_opid=self.active_operation_id,
            ) as writer:
                raw = run_recompute_plan(config, output_dir=output, writer=writer)
                artifacts = [
                    str(output / "scale_recompute_plan.json"),
                    str(output / "scale_recompute_plan.md"),
                ]
                manifest = manifest_base(
                    tool_id=contract.tool_id,
                    approval_id=str(arguments["approval"]["approval_id"]),
                )
                manifest.update({
                    "validation_id": validation_id,
                    "config_sha256": _sha256_file(config),
                    "store_snapshot_sha256": _scale_store_snapshot_sha256(summarize_store(store), store),
                    "resource_plan": resource_plan,
                    "data_label": _data_label(arguments),
                    "hidden_truth_included": False,
                })
                complete_manifest(
                    manifest,
                    status="completed" if ready else "partial",
                    artifacts=artifacts,
                    writer=writer,
                )
                manifest_path = write_execution_manifest(
                    output,
                    manifest,
                    filename="mcp_validation_manifest.json",
                    writer=writer,
                )
                return {
                    "status": "validated" if ready else "blocked",
                    "validation_id": validation_id if ready else "",
                    "execution_ready": ready,
                    "execution_authorized": False,
                    "selection": {"selected_object_count": len(selected)},
                    "work_counts": _scale_work_counts(raw),
                    "resource_plan": resource_plan,
                    "artifacts": artifacts,
                    "manifest_path": str(manifest_path),
                    "hidden_truth_visible": False,
                    **self._m5_policy_projection(contract, arguments),
                    "non_claims": [
                        "Validation does not execute propagation, screening, refinement, or ONP.",
                        "Scale and ONP comparisons are not collision probability or flight-safety authority.",
                    ],
                }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["execution_ready"] else "partial",
            evidence=lambda result: {
                "complete": result["execution_ready"], "empty": False, "truncated": False
            },
        )


    def _run_scale_refresh(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            config, _cfg, store, selected = self._bounded_scale_request(arguments)
            resource_plan = {
                "resource_profile": M5_RESOURCE_PROFILE,
                "workers": 1,
                "selected_object_count": len(selected),
                "max_selected_objects": M5_SCALE_MAX_SELECTED_OBJECTS,
                "action": "proceed" if len(selected) <= M5_SCALE_MAX_SELECTED_OBJECTS else "refuse",
            }
            self._enforce_approval_resource_ceiling(resource_plan)
            output = self.path_policy.resolve_write(arguments["output_dir"])
            exact_id = _scale_refresh_validation_id(
                config=config, store=store, selected_object_ids=selected,
                data_label=_data_label(arguments),
            )
            if str(arguments["validation_id"]) != exact_id:
                raise PermissionError("The Scale validation id does not match the exact config and store snapshot.")
            with SecureWriteRoot.open(
                output,
                approved_roots=self.path_policy.write_roots,
                create=True,
                require_empty=True,
                active_opid=self.active_operation_id,
            ) as writer:
                with bind_output_writer(writer):
                    execution_store = _copy_scale_store(
                        store,
                        output / "mcp_execution_store.sqlite",
                        writer=writer,
                    )
                    exact_id_after_backup = _scale_refresh_validation_id(
                        config=config, store=store, selected_object_ids=selected,
                        data_label=_data_label(arguments),
                    )
                    if exact_id_after_backup != exact_id:
                        raise PermissionError("The Scale store changed while creating the execution snapshot.")
                    generated_config = _write_scale_output_config(
                        config, output, store_path=execution_store, writer=writer
                    )
                    manifest = manifest_base(
                        tool_id=contract.tool_id,
                        approval_id=str(arguments["approval"]["approval_id"]),
                    )
                    manifest.update({
                        "validation_id": exact_id,
                        "config_sha256": _sha256_file(config),
                        "store_snapshot_sha256": _scale_store_snapshot_sha256(summarize_store(store), store),
                        "execution_store_sha256": writer.sha256("mcp_execution_store.sqlite"),
                        "execution_uses_isolated_snapshot": True,
                        "workers": 1,
                        "data_label": _data_label(arguments),
                        "hidden_truth_included": False,
                    })
                    manifest_path = output / "mcp_execution_manifest.json"
                    write_execution_manifest(output, manifest, writer=writer)
                    check = cancellation_callback(cancel_event)
                    try:
                        check()
                        raw = self._execute_scale_refresh(
                            generated_config,
                            workers=1,
                            step_callback=_m5_step_callback(cancel_event, progress),
                            writer=writer,
                        )
                        check()
                        artifacts = _scale_refresh_artifacts(output, raw, writer=writer)
                        status = str(raw.get("status", "partial"))
                        complete_manifest(
                            manifest,
                            status=status,
                            artifacts=artifacts,
                            writer=writer,
                        )
                        write_execution_manifest(output, manifest, writer=writer)
                        return self._scale_refresh_projection(
                            contract, arguments, raw, exact_id, artifacts, manifest_path
                        )
                    except MCPExecutionCancelled:
                        artifacts = _scale_refresh_artifacts(output, {}, writer=writer)
                        complete_manifest(
                            manifest,
                            status="cancelled",
                            cancelled=True,
                            artifacts=artifacts,
                            writer=writer,
                            error_type="MCPExecutionCancelled",
                        )
                        write_execution_manifest(output, manifest, writer=writer)
                        return self._scale_refresh_projection(
                            contract, arguments, {"status": "cancelled"}, exact_id, artifacts, manifest_path
                        )
                    except Exception as exc:
                        artifacts = _scale_refresh_artifacts(output, {}, writer=writer)
                        complete_manifest(
                            manifest,
                            status="failed",
                            artifacts=artifacts,
                            writer=writer,
                            error_type=type(exc).__name__,
                        )
                        write_execution_manifest(output, manifest, writer=writer)
                        raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"] if result["status"] in {"completed", "failed"} else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed", "empty": not bool(result["steps"]),
                "truncated": False,
            },
        )


    def _inspect_scale_refresh(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            workflow_path = self.path_policy.resolve_read_child(
                output, "scale_refresh_to_onp_workflow.json", kind="file"
            )
            manifest_path = self.path_policy.resolve_read_child(
                output, "mcp_execution_manifest.json", kind="file", required=False
            )
            require_file_size(workflow_path, maximum=MAX_MANIFEST_BYTES)
            raw = _load_json_object(workflow_path)
            manifest = _load_optional_manifest(manifest_path)
            manifest["artifacts_complete"] = _manifest_artifacts_valid(manifest, output)
            projected = self._scale_refresh_projection(
                contract, arguments, raw, str(manifest.get("validation_id", "")),
                _scale_refresh_artifacts(output, raw), manifest_path,
            )
            projected["manifest"] = _manifest_projection(manifest)
            if projected["status"] == "completed" and not manifest.get("artifacts_complete", False):
                projected["status"] = "partial"
            return projected

        return self._envelope(
            contract=contract, arguments=arguments, operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
        )


    def _bounded_scale_request(self, arguments: dict[str, Any]) -> tuple[Path, Any, Path, tuple[str, ...]]:
        config = self.path_policy.resolve_read(arguments["config_path"], kind="file")
        require_file_size(config, maximum=MAX_MANIFEST_BYTES)
        cfg = load_scale_planner_config(config)
        if cfg.mode != "run":
            raise ValueError("Scale refresh requires scale_catalog.mode: run.")
        if not cfg.onp_handoff_enabled:
            raise ValueError("Scale refresh to ONP requires scale_catalog.onp_handoff.enabled: true.")
        store = self.path_policy.resolve_read(cfg.store_path, kind="file")
        require_file_size(store, maximum=MAX_REVIEW_STORE_BYTES)
        selected = tuple(resolve_selected_object_ids(cfg))
        if len(selected) > M5_SCALE_MAX_SELECTED_OBJECTS:
            raise ValueError("The Scale selection exceeds the bounded MCP limit.")
        return config, cfg, store, selected


    def _scale_refresh_projection(
        self, contract: ToolContract, arguments: dict[str, Any], raw: dict[str, Any],
        validation_id: str, artifacts: list[str], manifest_path: Path,
    ) -> dict[str, Any]:
        status = str(raw.get("status", "partial"))
        if status not in {"completed", "partial", "failed", "cancelled"}:
            status = "partial"
        return {
            "status": status,
            "validation_id": validation_id,
            "workflow": {"type": raw.get("workflow_type"), "name": raw.get("workflow_name")},
            "steps": [
                {"name": row.get("name"), "status": row.get("status"), "summary": row.get("summary", {})}
                for row in list(raw.get("steps", []) or [])
            ],
            "final_health": dict(raw.get("final_health", {}) or {}),
            "comparison": dict(raw.get("comparison", {}) or {}),
            "artifacts": artifacts,
            "manifest_path": str(manifest_path),
            "hidden_truth_visible": False,
            **self._m5_policy_projection(contract, arguments),
            "non_claims": list(raw.get("non_claims", []) or []) or [
                "Scale and ONP differences are model comparisons, not operational decision authority."
            ],
        }


    @staticmethod
    def _m5_policy_projection(contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        features = list(_required_entitlements(contract))
        return {
            "data_label": _data_label(arguments),
            "entitlement": {
                "verified": bool(features),
                "feature": features[0] if features else "",
                "features": features,
            },
        }


def _required_entitlements(contract: ToolContract) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item
            for item in (contract.required_entitlement, *contract.required_entitlements)
            if item
        )
    )


def _data_label(arguments: dict[str, Any]) -> str:
    return str(dict(arguments.get("handling", {}) or {}).get("data_label", ""))


def _scale_refresh_validation_id(
    *, config: Path, store: Path, selected_object_ids: tuple[str, ...], data_label: str
) -> str:
    payload = json.dumps(
        {
            "config_sha256": _sha256_file(config),
            "store_snapshot_sha256": _scale_store_snapshot_sha256(summarize_store(store), store),
            "selected_object_ids": list(selected_object_ids),
            "resource_profile": M5_RESOURCE_PROFILE,
            "workers": 1,
            "data_label": data_label,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"oel-m5-scale-refresh-v1:{hashlib.sha256(payload).hexdigest()}"


def _write_scale_output_config(
    source: Path,
    output: Path,
    *,
    store_path: Path,
    writer: OutputWriter | None = None,
) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("scale_catalog"), dict):
        raise ValueError("Scale refresh config must contain a scale_catalog mapping.")
    planner = dict(raw["scale_catalog"].get("planner", {}) or {})
    planner["output_dir"] = str(output)
    raw["scale_catalog"]["planner"] = planner
    store = dict(raw["scale_catalog"].get("store", {}) or {})
    store["path"] = str(store_path)
    raw["scale_catalog"]["store"] = store
    handoff_key = "onp_handoff" if "onp_handoff" in raw["scale_catalog"] else "hpop_handoff"
    handoff = dict(raw["scale_catalog"].get(handoff_key, {}) or {})
    handoff["output_dir"] = str(output / "handoff")
    raw["scale_catalog"][handoff_key] = handoff
    sensitivity = dict(raw["scale_catalog"].get("sensitivity", {}) or {})
    sensitivity["output_dir"] = str(output / "sensitivity")
    raw["scale_catalog"]["sensitivity"] = sensitivity
    monte_carlo = dict(raw["scale_catalog"].get("monte_carlo", {}) or {})
    monte_carlo["output_dir"] = str(output / "monte_carlo")
    raw["scale_catalog"]["monte_carlo"] = monte_carlo
    target = output / "mcp_scale_refresh_config.yaml"
    serialized = yaml.safe_dump(raw, sort_keys=False).encode("utf-8")
    if writer is None:
        target.write_bytes(serialized)
        return target
    relative = target.relative_to(Path(writer.logical_root))
    writer.write_bytes(relative, serialized)
    # Config loading is path-facing; return the physical descriptor path while
    # the YAML itself retains logical output/store paths for producer metadata.
    return Path(writer.root) / relative


def _copy_scale_store(
    source: Path,
    target: Path,
    *,
    writer: OutputWriter | None = None,
) -> Path:
    if writer is not None:
        logical_root = Path(writer.logical_root)
        relative = target.relative_to(logical_root)
        try:
            writer.read_bytes(relative, max_bytes=1)
        except FileNotFoundError:
            pass
        except ValueError as exc:
            raise FileExistsError("The Scale execution-store snapshot must be new.") from exc
        else:
            raise FileExistsError("The Scale execution-store snapshot must be new.")
        staging = writer.create_staging(prefix=".oel-scale-store-copy-")
        stage_path = staging.path / "snapshot.sqlite"
    else:
        if target.exists():
            raise FileExistsError("The Scale execution-store snapshot must be new.")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = None
        stage_path = target
    source_uri = f"file:{source.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(source_uri, uri=True) as source_connection:
            with sqlite3.connect(stage_path) as target_connection:
                source_connection.backup(target_connection)
                integrity = target_connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or str(integrity[0]).lower() != "ok":
            raise ValueError("The Scale execution-store snapshot failed its integrity check.")
        if writer is not None:
            writer.publish_file(relative, stage_path)
            return target
        return target
    finally:
        if staging is not None and not staging._closed:
            staging.remove_tree()


def _scale_refresh_artifacts(
    output: Path,
    raw: dict[str, Any],
    *,
    writer: OutputWriter | None = None,
) -> list[str]:
    root = output.expanduser()
    physical_root = Path(_physical_writer_path(root, writer))
    candidates = [
        output / "scale_refresh_to_onp_workflow.json",
        output / "scale_refresh_to_onp_workflow.md",
        output / "scale_refresh_to_onp_comparison.json",
        output / "mcp_scale_refresh_config.yaml",
        output / "mcp_execution_store.sqlite",
    ]
    for value in dict(raw.get("artifacts", {}) or {}).values():
        if not isinstance(value, str) or not value:
            continue
        candidate = Path(value).expanduser()
        candidates.append(candidate if candidate.is_absolute() else root / candidate)
    available: set[Path] | None = None
    if writer is not None and callable(getattr(writer, "list_files", None)):
        logical_root = Path(writer.logical_root)
        available = {logical_root / relative for relative in writer.list_files()}
    contained: list[str] = []
    for path in candidates:
        candidate = path.expanduser()
        if available is not None:
            if candidate in available:
                contained.append(str(candidate))
            continue
        physical = Path(_physical_writer_path(candidate, writer))
        try:
            physical.relative_to(physical_root)
        except ValueError:
            continue
        if physical.is_file() and not physical.is_symlink():
            contained.append(str(candidate))
    return list(dict.fromkeys(contained))


def _m5_step_callback(cancel_event: Event | None, progress: Any | None) -> Callable[[int, int, str], None]:
    check = cancellation_callback(cancel_event)

    def callback(completed: int, total: int, message: str) -> None:
        check()
        if progress:
            progress(completed, total, message)

    return callback


def _dynamics_od_artifacts(
    output: Path,
    raw: dict[str, Any],
    *,
    writer: OutputWriter | None = None,
) -> list[str]:
    """Collect the bounded OD evidence set without enumerating the root by path."""

    if writer is not None and callable(getattr(writer, "list_files", None)):
        logical_root = Path(writer.logical_root)
        return [
            str(logical_root / relative)
            for relative in writer.list_files()
            if relative != Path("mcp_execution_manifest.json")
        ]

    known = (
        "fitted_mission_input_packet.json",
        "estimated_parameters.json",
        "materialized_fit_config.yaml",
        "materialized_prediction_config.yaml",
        "fit_residuals.csv",
        "holdout_errors.csv",
        "fit_position_residuals.png",
        "holdout_position_errors.png",
        "od_fit_report.json",
        "od_fit_report.md",
        "state_estimate_product.json",
        "state_estimate_product_emission.json",
        "od_solution.json",
        "oel_workflow_evidence.json",
    )
    candidates: list[Path] = [output / name for name in known]

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child, key)
        elif key.endswith("_path") and isinstance(value, (str, Path)):
            candidate = Path(value).expanduser()
            candidates.append(candidate if candidate.is_absolute() else output / candidate)

    collect(raw)
    physical_root = Path(_physical_writer_path(output, writer))
    logical: list[str] = []
    for candidate in candidates:
        logical_candidate = candidate if candidate.is_absolute() else output / candidate
        physical_candidate = Path(_physical_writer_path(logical_candidate, writer))
        try:
            physical_candidate.relative_to(physical_root)
        except ValueError:
            continue
        if physical_candidate.is_file() and not physical_candidate.is_symlink():
            logical.append(str(logical_candidate))
    return list(dict.fromkeys(logical))


def _physical_writer_path(path: str | Path, writer: OutputWriter | None = None) -> str:
    """Map a logical producer path to the bound physical root when needed."""

    candidate = Path(path).expanduser()
    if writer is None:
        return str(candidate)
    logical_root = Path(writer.logical_root).expanduser()
    physical_root = Path(writer.root).expanduser()
    try:
        return str(physical_root / candidate.relative_to(logical_root))
    except ValueError:
        return str(candidate)


def _scale_work_counts(raw: dict[str, Any]) -> dict[str, int]:
    return {
        "recompute_chunks": int(dict(raw.get("propagation", {}) or {}).get("recompute_chunk_count", 0) or 0),
        "rescreen_work": int(dict(raw.get("screening", {}) or {}).get("total_rescreen_work_count", 0) or 0),
        "pending_refinements": int(dict(raw.get("candidates", {}) or {}).get("pending_refinement_count", 0) or 0),
        "pending_handoff_packets": int(dict(raw.get("hpop_handoff", {}) or {}).get("pending_packet_count", 0) or 0),
    }


def _scale_job_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: row.get(key) for key in ("job_type", "count", "status", "reason") if key in row}
        for row in rows
    ]


def _scale_store_snapshot_sha256(summary: Any, store: Path) -> str:
    del summary  # The identity binds the complete transactional store, not coarse counters.
    uri = f"file:{store.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("BEGIN")
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check != ("ok",):
            raise ValueError(f"Scale store quick_check failed: {quick_check!r}")
        digest = hashlib.sha256()
        for statement in connection.iterdump():
            digest.update(statement.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trajectory_resource_plan(problem: TrajectoryOptimizationProblem) -> dict[str, Any]:
    duration_bound_s = sum(
        float(segment.get("duration_s", dict(segment.get("stop", {}) or {}).get("max_duration_s", 0.0)))
        for segment in problem.segments
    )
    evaluation_budget = int(problem.solver.get("evaluation_budget", 0))
    robustness_samples = int(problem.robustness.get("sample_count", 0))
    derived_steps = math.ceil(duration_bound_s / float(problem.propagation["step_s"]))
    derived_steps *= evaluation_budget + robustness_samples + 1
    action = (
        "proceed"
        if evaluation_budget <= 20_000 and derived_steps <= 20_000_000 and len(problem.variables) <= 128
        else "refuse"
    )
    return {
        "resource_profile": "laptop-safe",
        "segment_count": len(problem.segments),
        "variable_count": len(problem.variables),
        "constraint_count": len(problem.constraints),
        "evaluation_budget": evaluation_budget,
        "solver_iteration_limit": int(problem.solver.get("max_iterations", 0)),
        "robustness_sample_count": robustness_samples,
        "derived_propagation_steps": derived_steps,
        "max_derived_propagation_steps": 20_000_000,
        "action": action,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The bounded Pro evidence artifact must contain a JSON object.")
    return value


def _load_optional_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "partial", "artifacts_complete": False, "cancelled": False}
    require_file_size(path, maximum=MAX_MANIFEST_BYTES)
    return _load_json_object(path)


def _manifest_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "status",
        "artifacts_complete",
        "cancelled",
        "validation_id",
        "resource_profile",
        "config_sha256",
        "store_snapshot_sha256",
        "data_label",
        "started_utc",
        "completed_utc",
    )
    return {key: manifest.get(key) for key in allowed if key in manifest}


def _manifest_artifacts_valid(manifest: dict[str, Any], output: Path) -> bool:
    if not bool(manifest.get("artifacts_complete", False)):
        return False
    records = list(manifest.get("artifact_records", []) or [])
    artifacts = list(manifest.get("artifacts", []) or [])
    if not records or len(records) != len(artifacts):
        return False
    expected_paths = {str(path) for path in artifacts}
    seen: set[str] = set()
    for raw_record in records:
        record = dict(raw_record or {})
        raw_path = str(record.get("path", "") or "")
        path = Path(raw_path)
        try:
            path.resolve().relative_to(output.resolve())
        except ValueError:
            return False
        if raw_path not in expected_paths or raw_path in seen or not path.is_file() or path.is_symlink():
            return False
        if int(record.get("bytes", -1)) != path.stat().st_size:
            return False
        if str(record.get("sha256", "")) != _sha256_file(path):
            return False
        seen.add(raw_path)
    return seen == expected_paths
