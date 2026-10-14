from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers, require_file_size
from integrations.oel_mcp.contracts import (
    MAX_MANIFEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_REVIEW_RESULT_BYTES,
    MAX_REVIEW_STORE_BYTES,
    MAX_REVIEW_VALUE_BYTES,
    ToolContract,
)
from integrations.oel_mcp.cr3bp_handlers import call_cr3bp_tool
from integrations.oel_mcp.execution import (
    ExecutionApprovalPolicy,
    MCPExecutionCancelled,
    bounded_config_path_policy,
    cancellation_callback,
    complete_manifest,
    ensure_new_output_dir,
    manifest_base,
    prepare_scenario,
    require_safe_resource_estimate,
    resource_estimate,
    safe_artifact_id,
    validate_prepared_scenario,
    write_execution_manifest,
    write_materialized_config,
)
from integrations.oel_mcp.fsw_authoring_handlers import call_fsw_authoring_tool
from integrations.oel_mcp.public_registry import public_contract_map
from integrations.oel_mcp.recovery import annotate_error
from integrations.oel_mcp.reporting import MAX_REPORT_SOURCE_BYTES
from integrations.oel_mcp.reporting import audit_report as audit_report_artifacts
from integrations.oel_mcp.reporting import prepare_report_packet as prepare_report_packet_artifacts
from sim.agent_task.plot_recipes import get_plot_recipe
from sim.agent_task.recipes import get_recipe, is_public_mcp_recipe
from sim.agent_task.runner import AgentTaskCancelled, compare_outputs, create_plot, inspect_output, run_recipe
from sim.ccsds import inspect_cdm, inspect_odm, inspect_oem, inspect_tdm
from sim.execution import run_simulation_config_file
from sim.frame_time import (
    FrameTimeError,
    FrameTransformContext,
    TimeScale,
    audit_eop_series,
    epoch_conversion_receipt,
    frame_transform_receipt,
    load_iers_eop,
    parse_epoch,
    transform_cartesian_state,
    transform_covariance,
)
from sim.handoff import (
    compare_handoff,
    emit_scenario_overlay,
    export_completed_run_snapshot,
    export_completed_run_state,
    export_maneuver_detection_product,
    inspect_path,
    load_interchange_document,
    load_scenario_overlay,
    materialize_onp,
    materialize_scenario_patch,
    materialize_snapshot_onp,
)
from sim.reporting import build_maneuver_readiness_packet
from sim.review import (
    ReviewWorkspace,
    animation_spec_from_mapping,
    plan_review_animation,
    plan_review_plot,
    plot_spec_from_mapping,
)
from sim.review import render_review_animation as render_planned_review_animation
from sim.review import render_review_plot as render_planned_review_plot
from sim.study import compare_study_bundles, inspect_study_bundle, replay_study_bundle


class PublicOELMCPHandlers(BaseOELMCPHandlers):
    def __init__(
        self,
        *,
        profile: str = "public_local",
        read_roots: tuple[str | Path, ...] | None = None,
        write_roots: tuple[str | Path, ...] | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        approval_policy: ExecutionApprovalPolicy | None = None,
    ) -> None:
        super().__init__(
            profile=profile,
            contracts=public_contract_map(profile),
            read_roots=read_roots,
            write_roots=write_roots,
            max_response_bytes=max_response_bytes,
            approval_policy=approval_policy,
        )

    def inspect_run(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_run.v1", arguments)

    def query_review(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.query_review.v1", arguments)

    def plan_run(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plan_run.v1", arguments)

    def validate_scenario(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.validate_scenario.v1", arguments)

    def run_scenario(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.run_scenario.v1", arguments)

    def compare_runs(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.compare_runs.v1", arguments)

    def plot_evidence(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plot_evidence.v1", arguments)

    def plan_review_plot(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plan_review_plot.v1", arguments)

    def render_review_plot(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.render_review_plot.v2", arguments)

    def plan_review_animation(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plan_review_animation.v1", arguments)

    def render_review_animation(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.render_review_animation.v1", arguments)

    def run_agent_task(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.run_agent_task.v1", arguments)

    def prepare_report_packet(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.prepare_report_packet.v1", arguments)

    def audit_report(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.audit_report.v1", arguments)

    def inspect_handoff(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_handoff.v1", arguments)

    def export_run_product(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.export_run_product.v1", arguments)

    def emit_scenario_overlay(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.emit_scenario_overlay.v1", arguments)

    def materialize_onp_handoff(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.materialize_onp_handoff.v1", arguments)

    def materialize_scenario_patch(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.materialize_scenario_patch.v1", arguments)

    def compare_handoff(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.compare_handoff.v1", arguments)

    def assess_maneuver_readiness(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.assess_maneuver_readiness.v1", arguments)

    def inspect_study(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_study.v1", arguments)

    def replay_study(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.replay_study.v1", arguments)

    def compare_studies(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.compare_studies.v1", arguments)

    def inspect_ccsds(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.inspect_ccsds.v1", arguments)

    def convert_frame_time(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.convert_frame_time.v1", arguments)

    def plan_cr3bp_render(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.plan_cr3bp_render.v1", arguments)

    def render_cr3bp(self, **arguments: Any) -> dict[str, Any]:
        return self.call("oel.render_cr3bp.v1", arguments)

    def _call_contract(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        cr3bp_result = call_cr3bp_tool(
            self, contract, arguments, cancel_event=cancel_event, progress=progress,
        )
        if cr3bp_result is not None:
            return cr3bp_result
        fsw_result = call_fsw_authoring_tool(self, contract, arguments)
        if fsw_result is not None:
            return fsw_result
        if contract.tool_id == "oel.inspect_run.v1":
            return self._inspect_run(contract, arguments)
        if contract.tool_id == "oel.query_review.v1":
            return self._query_review(contract, arguments)
        if contract.tool_id == "oel.plan_run.v1":
            return self._plan_run(contract, arguments)
        if contract.tool_id == "oel.validate_scenario.v1":
            return self._validate_scenario(contract, arguments)
        if contract.tool_id == "oel.run_scenario.v1":
            return self._run_scenario(contract, arguments, cancel_event=cancel_event, progress=progress)
        if contract.tool_id == "oel.compare_runs.v1":
            return self._compare_runs(contract, arguments)
        if contract.tool_id == "oel.plot_evidence.v1":
            return self._plot_evidence(contract, arguments)
        if contract.tool_id == "oel.plan_review_plot.v1":
            return self._plan_review_plot(contract, arguments)
        if contract.tool_id == "oel.render_review_plot.v2":
            return self._render_review_plot(contract, arguments)
        if contract.tool_id == "oel.plan_review_animation.v1":
            return self._plan_review_animation(contract, arguments)
        if contract.tool_id == "oel.render_review_animation.v1":
            return self._render_review_animation(contract, arguments)
        if contract.tool_id == "oel.run_agent_task.v1":
            return self._run_agent_task(contract, arguments, cancel_event=cancel_event, progress=progress)
        if contract.tool_id == "oel.prepare_report_packet.v1":
            return self._prepare_report_packet(contract, arguments)
        if contract.tool_id == "oel.audit_report.v1":
            return self._audit_report(contract, arguments)
        if contract.tool_id == "oel.inspect_handoff.v1":
            return self._inspect_handoff(contract, arguments)
        if contract.tool_id == "oel.export_run_product.v1":
            return self._export_run_product(contract, arguments)
        if contract.tool_id == "oel.emit_scenario_overlay.v1":
            return self._emit_scenario_overlay(contract, arguments)
        if contract.tool_id == "oel.materialize_onp_handoff.v1":
            return self._materialize_onp_handoff(contract, arguments)
        if contract.tool_id == "oel.materialize_scenario_patch.v1":
            return self._materialize_scenario_patch(contract, arguments)
        if contract.tool_id == "oel.compare_handoff.v1":
            return self._compare_handoff(contract, arguments)
        if contract.tool_id == "oel.assess_maneuver_readiness.v1":
            return self._assess_maneuver_readiness(contract, arguments)
        if contract.tool_id == "oel.inspect_study.v1":
            return self._inspect_study(contract, arguments)
        if contract.tool_id == "oel.replay_study.v1":
            return self._replay_study(contract, arguments)
        if contract.tool_id == "oel.compare_studies.v1":
            return self._compare_studies(contract, arguments)
        if contract.tool_id == "oel.inspect_ccsds.v1":
            return self._inspect_ccsds(contract, arguments)
        if contract.tool_id == "oel.convert_frame_time.v1":
            return self._convert_frame_time(contract, arguments)
        raise PermissionError("Tool is not available in this deployment profile.")

    def _inspect_study(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            bundle = self.path_policy.resolve_read(arguments["bundle_dir"], kind="directory")
            _authorize_study_bundle(self.path_policy, bundle)
            return inspect_study_bundle(bundle)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )

    def _replay_study(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            bundle = self.path_policy.resolve_read(arguments["bundle_dir"], kind="directory")
            _authorize_study_bundle(self.path_policy, bundle)
            return replay_study_bundle(bundle)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )

    def _compare_studies(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            left = self.path_policy.resolve_read(arguments["left_bundle_dir"], kind="directory")
            right = self.path_policy.resolve_read(arguments["right_bundle_dir"], kind="directory")
            _authorize_study_bundle(self.path_policy, left)
            _authorize_study_bundle(self.path_policy, right)
            return compare_study_bundles(left, right)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )

    def _inspect_ccsds(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        inspectors = {"oem": inspect_oem, "odm": inspect_odm, "tdm": inspect_tdm, "cdm": inspect_cdm}

        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["path"], kind="file")
            require_file_size(source, maximum=MAX_MANIFEST_BYTES)
            product_kind = str(arguments["product_kind"])
            return {
                "status": "inspected",
                "product_kind": product_kind,
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "inspection": inspectors[product_kind](source),
                "execution_occurred": False,
                "non_claims": [
                    "CCSDS parsing and profile inspection do not certify upstream navigation accuracy.",
                    "Inspection does not execute a scenario or authorize use of the message for operations.",
                ],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )

    def _convert_frame_time(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            operation_id = str(arguments["operation"])
            if operation_id == "convert_epoch":
                _require_argument_fields(arguments, operation_id, "epoch", "from_scale", "to_scale")
                dut1_s = arguments.get("dut1_s")
                epoch = parse_epoch(arguments["epoch"], arguments["from_scale"], dut1_s=dut1_s)
                receipt = epoch_conversion_receipt(epoch, arguments["to_scale"], dut1_s=dut1_s)
                return _frame_time_projection(operation_id, result=dict(receipt["output"]), receipt=receipt)
            if operation_id == "inspect_eop":
                _require_argument_fields(arguments, operation_id, "eop_path")
                source = self.path_policy.resolve_read(arguments["eop_path"], kind="file")
                require_file_size(source, maximum=MAX_MANIFEST_BYTES)
                series = load_iers_eop(source, source_format=str(arguments.get("eop_format", "auto")))
                audit = audit_eop_series(
                    series,
                    as_of=_parse_utc_datetime(arguments.get("as_of")),
                    max_observed_age_days=float(arguments.get("max_observed_age_days", 45.0)),
                )
                return _frame_time_projection(
                    operation_id,
                    status="inspected",
                    result=audit,
                    receipt=audit,
                    eop_source=series.receipt(),
                )
            _require_argument_fields(arguments, operation_id, "epoch", "time_scale", "source_frame", "target_frame")
            context, eop_receipt = self._frame_time_context(arguments)
            if operation_id == "transform_state":
                _require_argument_fields(arguments, operation_id, "position_km", "velocity_km_s")
                position, velocity = transform_cartesian_state(
                    arguments["position_km"],
                    arguments["velocity_km_s"],
                    arguments["source_frame"],
                    arguments["target_frame"],
                    context=context,
                )
                result = {"position_km": position.tolist(), "velocity_km_s": velocity.tolist()}
            else:
                _require_argument_fields(arguments, operation_id, "covariance")
                result = {
                    "covariance": transform_covariance(
                        arguments["covariance"],
                        arguments["source_frame"],
                        arguments["target_frame"],
                        context=context,
                    ).tolist()
                }
            receipt = frame_transform_receipt(arguments["source_frame"], arguments["target_frame"], context=context)
            return _frame_time_projection(operation_id, result=result, receipt=receipt, eop_source=eop_receipt)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda _result: "completed",
            evidence=lambda _result: {"complete": True, "empty": False, "truncated": False},
        )

    def _frame_time_context(self, arguments: dict[str, Any]) -> tuple[FrameTransformContext, dict[str, Any] | None]:
        scale = TimeScale(str(arguments["time_scale"]).upper())
        series = None
        if arguments.get("eop_path"):
            source = self.path_policy.resolve_read(arguments["eop_path"], kind="file")
            require_file_size(source, maximum=MAX_MANIFEST_BYTES)
            series = load_iers_eop(source, source_format=str(arguments.get("eop_format", "auto")))
        if scale is TimeScale.UT1 and series is None:
            raise FrameTimeError("UT1 input requires an epoch-matched EOP source.")
        if scale is TimeScale.UT1:
            provisional = parse_epoch(arguments["epoch"], scale, dut1_s=0.0)
            epoch_eop = series.sample(provisional)
            epoch = parse_epoch(arguments["epoch"], scale, dut1_s=epoch_eop.dut1_s)
        else:
            epoch = parse_epoch(arguments["epoch"], scale)
        eop = None if series is None else series.sample(epoch)
        return FrameTransformContext(epoch=epoch, earth_orientation=eop), None if series is None else series.receipt()

    def _inspect_run(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=False)
            db_path = self.path_policy.resolve_read_child(
                output, "review", "run.sqlite", kind="file", required=False
            )
            if db_path.is_file():
                _require_review_store_budget(db_path)
            raw = inspect_output(
                output,
                query_names=arguments.get("query_names"),
                max_rows=int(arguments.get("max_rows", 50)),
                write_packet=False,
                max_value_bytes=MAX_REVIEW_VALUE_BYTES,
                max_result_bytes=min(MAX_REVIEW_RESULT_BYTES, self.max_response_bytes),
            )
            manifest_problem = _incomplete_manifest_status(output)
            if manifest_problem:
                raw["status"] = "failed"
                raw.setdefault("failure_hints", []).append(manifest_problem)
            review = dict(raw.get("review", {}) or {})
            execution_provenance = _execution_provenance(output)
            artifact_summary = dict(raw.get("artifact_summary", {}) or {})
            artifact_summary.update(
                {
                    "mcp_manifest_artifact_count": execution_provenance["artifact_count"],
                    "mcp_manifest_artifacts_complete": execution_provenance["artifacts_complete"],
                }
            )
            evidence_summary = dict(raw.get("evidence_summary", {}) or {})
            evidence_summary["mcp_execution_complete"] = bool(
                execution_provenance["available"]
                and execution_provenance["status"] == "completed"
                and execution_provenance["artifacts_complete"]
            )
            return {
                "output_dir": str(output),
                "status": str(raw.get("status", "partial")),
                "evidence_summary": evidence_summary,
                "review": {
                    key: review[key]
                    for key in ("db_path", "tables", "query_summary", "queries", "error")
                    if key in review
                },
                "artifact_summary": artifact_summary,
                "execution_provenance": execution_provenance,
                "failure_hints": [dict(item) for item in list(raw.get("failure_hints", []) or [])],
                "caveats": [str(item) for item in list(raw.get("caveats", []) or [])],
                "freshness": {
                    "current_config_reproduction_verified": False,
                    "content_bound_execution_recorded": bool(execution_provenance["validation_id"]),
                    "meaning": (
                        "Inspection verifies the current output contents and reports their MCP execution manifest; "
                        "it does not rerun the source configuration."
                    ),
                },
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "completed"
                and not bool(dict(result.get("review", {}) or {}).get("error")),
                "empty": False,
                "truncated": bool(
                    dict(dict(result.get("review", {}) or {}).get("query_summary", {}) or {}).get("truncated", 0)
                ),
            },
        )

    def _query_review(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            path = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, path, require_store=True)
            workspace = ReviewWorkspace.open(path)
            _require_review_store_budget(workspace.db_path)
            result = workspace.query(
                str(arguments["sql"]),
                max_rows=int(arguments.get("max_rows", 100)),
                max_vm_steps=int(arguments.get("max_vm_steps", 250_000)),
                max_value_bytes=MAX_REVIEW_VALUE_BYTES,
                max_result_bytes=min(MAX_REVIEW_RESULT_BYTES, self.max_response_bytes),
            )
            return {
                "output_dir": str(workspace.output_dir),
                "review_store": "review/run.sqlite",
                "sql": str(arguments["sql"]),
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "empty_result": result.row_count == 0,
                "empty_result_semantics": "No matching review evidence was recorded for this query.",
                "truncated": result.truncated,
                "units_semantics": "Units are carried by selected columns and saved-query metadata; no units are inferred.",
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "partial" if result["truncated"] else "completed",
            evidence=lambda result: {
                "complete": not result["truncated"],
                "empty": result["empty_result"],
                "truncated": result["truncated"],
            },
        )

    def _prepared(self, arguments: dict[str, Any]) -> Any:
        return prepare_scenario(
            config_path=arguments["config_path"],
            output_dir=arguments["output_dir"],
            resource_profile=str(arguments.get("resource_profile", "laptop-safe")),
            path_policy=self.path_policy,
        )

    @staticmethod
    def _scenario_projection(prepared: Any) -> dict[str, Any]:
        cfg = prepared.config
        return {
            "scenario_name": cfg.scenario_name,
            "scenario_description": cfg.scenario_description,
            "duration_s": float(cfg.simulator.duration_s),
            "dt_s": float(cfg.simulator.dt_s),
            "output_dir": str(prepared.output_dir),
            "resource_profile": prepared.resource_profile,
        }

    @staticmethod
    def _identity_projection(prepared: Any, *, validation_id: str = "") -> dict[str, Any]:
        return {
            "source_config_sha256": prepared.raw_sha256,
            "normalized_config_sha256": prepared.normalized_sha256,
            "validation_id": validation_id,
        }

    def _plan_run(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(prepared, path_policy=self.path_policy, trust_plugins=False)
            return {
                "scenario": self._scenario_projection(prepared),
                "identity": self._identity_projection(prepared),
                "safe_validation": validation["safe_validation"],
                "resource_estimate": resource_estimate(prepared),
                "phases": [
                    "safe_validate",
                    "trusted_validate",
                    "resource_preflight",
                    "operator_approval",
                    "deterministic_execute",
                    "inspect_evidence",
                ],
                "approval": {
                    "required_for_execution": True,
                    "configured_outside_model": True,
                    "environment": "OEL_MCP_EXECUTION_APPROVAL_IDS",
                },
                "execution_authorized": False,
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["safe_validation"].get("ok") else "partial",
            evidence=lambda result: {
                "complete": bool(result["safe_validation"].get("ok")),
                "empty": False,
                "truncated": False,
            },
        )

    def _validate_scenario(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(
                prepared,
                path_policy=self.path_policy,
                trust_plugins=bool(arguments.get("trust_plugins", False)),
            )
            estimate = resource_estimate(prepared)
            status = str(validation["status"])
            return {
                "status": status,
                "scenario": self._scenario_projection(prepared),
                "identity": self._identity_projection(prepared, validation_id=str(validation["validation_id"])),
                "safe_validation": validation["safe_validation"],
                "trusted_validation": validation["trusted_validation"],
                "resource_estimate": estimate,
                "execution_ready": bool(validation["execution_ready"]) and estimate["action"] != "refuse",
                "execution_authorized": False,
                "next_step": _validation_next_step(
                    validation=validation,
                    resource_action=str(estimate["action"]),
                    trust_plugins=bool(arguments.get("trust_plugins", False)),
                ),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                "completed" if result["status"] == "validated" and result["execution_ready"] else "partial"
            ),
            evidence=lambda result: {"complete": bool(result["execution_ready"]), "empty": False, "truncated": False},
        )

    def _run_scenario(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            prepared = self._prepared(arguments)
            validation = validate_prepared_scenario(prepared, path_policy=self.path_policy, trust_plugins=True)
            if not validation["execution_ready"] or arguments["validation_id"] != prepared.validation_id:
                raise annotate_error(
                    PermissionError("The trusted validation id does not match the exact execution-normalized config."),
                    "validation.stale",
                )
            estimate = resource_estimate(prepared)
            require_safe_resource_estimate(estimate)
            ensure_new_output_dir(prepared.output_dir)
            approval_id = str(arguments["approval"]["approval_id"])
            manifest = manifest_base(tool_id=contract.tool_id, approval_id=approval_id, prepared=prepared)
            manifest["resource_estimate"] = estimate
            manifest_path = write_execution_manifest(prepared.output_dir, manifest)
            materialized = prepared.output_dir / "mcp_execution_config.yaml"
            try:
                materialized = write_materialized_config(prepared)
                execution_path_policy = bounded_config_path_policy(
                    config_path=materialized,
                    path_policy=self.path_policy,
                )
                if progress:
                    progress(1, 3, "Validated and materialized the approved scenario.")
                payload = run_simulation_config_file(
                    materialized,
                    path_policy=execution_path_policy,
                    step_callback=cancellation_callback(cancel_event),
                )
                cancellation_callback(cancel_event)()
                if progress:
                    progress(2, 3, "Deterministic execution completed; collecting artifacts.")
                artifacts = _existing_artifacts(prepared.output_dir, materialized)
                complete_manifest(manifest, status="completed", artifacts=artifacts)
                write_execution_manifest(prepared.output_dir, manifest)
                if progress:
                    progress(3, 3, "Execution evidence manifest completed.")
                return {
                    "status": "completed",
                    "scenario": self._scenario_projection(prepared),
                    "identity": self._identity_projection(prepared, validation_id=prepared.validation_id),
                    "output_dir": str(prepared.output_dir),
                    "run": _run_projection(payload),
                    "artifacts": artifacts,
                    "manifest_path": str(manifest_path),
                }
            except MCPExecutionCancelled:
                artifacts = _existing_artifacts(prepared.output_dir, materialized)
                complete_manifest(
                    manifest,
                    status="cancelled",
                    cancelled=True,
                    artifacts=artifacts,
                    error_type="MCPExecutionCancelled",
                )
                write_execution_manifest(prepared.output_dir, manifest)
                return {
                    "status": "cancelled",
                    "scenario": self._scenario_projection(prepared),
                    "identity": self._identity_projection(prepared, validation_id=prepared.validation_id),
                    "output_dir": str(prepared.output_dir),
                    "run": {"cancelled": True},
                    "artifacts": artifacts,
                    "manifest_path": str(manifest_path),
                }
            except Exception as exc:
                complete_manifest(
                    manifest,
                    status="failed",
                    artifacts=_existing_artifacts(prepared.output_dir, materialized),
                    error_type=type(exc).__name__,
                )
                write_execution_manifest(prepared.output_dir, manifest)
                raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "completed" else "partial",
            evidence=lambda result: {"complete": result["status"] == "completed", "empty": False, "truncated": False},
        )

    def _compare_runs(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            base = self.path_policy.resolve_read(arguments["base_output_dir"], kind="directory")
            candidate = self.path_policy.resolve_read(arguments["candidate_output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, base, require_store=False)
            _authorize_review_inputs(self.path_policy, candidate, require_store=False)
            for output in (base, candidate):
                db = output / "review" / "run.sqlite"
                if db.is_file():
                    _require_review_store_budget(db)
                problem = _incomplete_manifest_status(output)
                if problem:
                    raise ValueError(
                        f"Comparison source execution is not complete: {problem['status']}."
                    )
            raw = compare_outputs(
                base,
                candidate,
                metric_names=arguments["metric_names"],
                max_rows=int(arguments.get("max_rows", 50)),
                max_value_bytes=MAX_REVIEW_VALUE_BYTES,
                max_result_bytes=min(MAX_REVIEW_RESULT_BYTES, self.max_response_bytes),
            )
            return {
                "status": raw["status"],
                "base_output_dir": str(base),
                "candidate_output_dir": str(candidate),
                "metric_names": raw["metric_names"],
                "metrics": raw["metrics"],
                "deltas": raw["deltas"],
                "metric_status": raw["metric_status"],
                "summary": raw["summary"],
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                result["status"] if result["status"] in {"completed", "failed"} else "partial"
            ),
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": not bool(result["metric_names"]),
                "truncated": False,
            },
        )

    def _plot_evidence(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            self.path_policy.resolve_write(output)
            recipe_id = str(arguments["recipe_id"])
            recipe = get_plot_recipe(recipe_id)
            if recipe is None or recipe.maturity != "supported":
                raise ValueError("Only supported allowlisted plot recipes are available.")
            artifact_id = safe_artifact_id(str(arguments["artifact_id"]))
            file_format = str(arguments.get("format", "png"))
            target = output / "review" / "mcp_plots" / f"{artifact_id}.{file_format}"
            suffix = 2
            while target.exists():
                target = output / "review" / "mcp_plots" / f"{artifact_id}_{suffix}.{file_format}"
                suffix += 1
            artifact_id = safe_artifact_id(target.stem)
            target = self.path_policy.resolve_write(target)
            artifact = create_plot(
                output,
                recipe_id,
                style_name=str(arguments.get("style", "oel_dark")),
                file_format=file_format,
                artifact_id=artifact_id,
                path=target,
            )
            provenance = _execution_provenance(output)
            summary_complete = False
            summary_path = output / "master_run_summary.json"
            if summary_path.is_file():
                require_file_size(summary_path, maximum=MAX_MANIFEST_BYTES)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(summary, dict):
                    summary_status = str(summary.get("status", "")).lower()
                    summary_complete = summary_status == "completed" or bool(
                        not summary_status
                        and (output / "review" / "run.sqlite").is_file()
                        and summary.get("scenario_name")
                        and summary.get("samples") is not None
                    )
            # An MCP manifest is the authoritative execution boundary when
            # present.  A legacy summary may fill in provenance only for
            # outputs that predate the manifest; it must not revive an
            # explicitly cancelled or incomplete MCP execution.
            source_complete = (
                bool(provenance["status"] == "completed" and provenance["artifacts_complete"])
                if provenance["available"]
                else summary_complete
            )
            qa = dict(artifact.get("qa", {}) or {})
            plot_complete = (
                source_complete
                and artifact.get("status") == "ok"
                and bool(artifact.get("path_exists"))
                and not bool(artifact.get("truncated"))
                and qa.get("automated_status") == "passed"
            )
            manifest = manifest_base(
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
            )
            manifest.update(
                {
                    "recipe_id": recipe_id,
                    "recipe_version": int(getattr(recipe, "recipe_version", 1)),
                    "query_sha256": hashlib.sha256(recipe.sql.encode("utf-8")).hexdigest(),
                    "qa": qa,
                }
            )
            manifest = complete_manifest(
                manifest,
                status="completed" if plot_complete else "partial",
                artifacts=[str(target)] if artifact.get("path_exists") else [],
            )
            manifest_path = write_execution_manifest(
                output / "review" / "mcp_plots",
                manifest,
                filename=f"{artifact_id}.manifest.json",
            )
            return {
                "status": "completed" if plot_complete else "partial",
                "output_dir": str(output),
                "recipe_id": recipe_id,
                "artifact": artifact,
                "manifest_path": str(manifest_path),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": False,
                "truncated": bool(result["artifact"].get("truncated")),
            },
        )

    def _plan_review_plot(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            spec = plot_spec_from_mapping(arguments, source="oel_mcp_review_plot_v2")
            return plan_review_plot(output, spec)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            evidence=lambda result: {
                "complete": not bool(result["truncated"]),
                "empty": False,
                "truncated": bool(result["truncated"]),
            },
        )

    def _render_review_plot(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            self.path_policy.resolve_write(output)
            artifact_id = safe_artifact_id(str(arguments["artifact_id"]))
            file_format = str(arguments.get("format", "png"))
            target = output / "review" / "mcp_plots" / f"{artifact_id}.{file_format}"
            if target.exists():
                raise ValueError("The planned plot target already exists; choose a new artifact_id and re-plan.")
            target = self.path_policy.resolve_write(target)
            spec = plot_spec_from_mapping(arguments, source="oel_mcp_review_plot_v2")
            artifact = render_planned_review_plot(
                output,
                spec,
                plot_plan_id=str(arguments["plot_plan_id"]),
                path=target,
            )
            artifact_payload = {
                "artifact_id": artifact.artifact_id,
                "path": str(artifact.path),
                "relative_path": artifact.relative_path,
                "path_exists": artifact.path.is_file(),
                "row_count": artifact.row_count,
                "truncated": artifact.truncated,
                "spec": asdict(artifact.spec),
                "qa": dict(artifact.qa),
            }
            complete = bool(
                artifact.path.is_file()
                and not artifact.truncated
                and artifact.qa.get("automated_status") == "passed"
            )
            manifest = manifest_base(
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
            )
            manifest.update(
                {
                    "plot_plan_id": str(arguments["plot_plan_id"]),
                    "artifact_id": artifact.artifact_id,
                    "query_sha256": hashlib.sha256(spec.sql.encode("utf-8")).hexdigest(),
                    "plot_spec": asdict(spec),
                    "qa": dict(artifact.qa),
                }
            )
            manifest = complete_manifest(
                manifest,
                status="completed" if complete else "partial",
                artifacts=[str(target)] if artifact.path.is_file() else [],
            )
            manifest_path = write_execution_manifest(
                output / "review" / "mcp_plots",
                manifest,
                filename=f"{artifact_id}.manifest.json",
            )
            return {
                "status": "completed" if complete else "partial",
                "output_dir": str(output),
                "plot_plan_id": str(arguments["plot_plan_id"]),
                "artifact": artifact_payload,
                "manifest_path": str(manifest_path),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": False,
                "truncated": bool(result["artifact"].get("truncated")),
            },
        )

    def _plan_review_animation(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            spec = animation_spec_from_mapping(arguments, source="oel_mcp_review_animation_v1")
            return plan_review_animation(output, spec)

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            evidence=lambda result: {
                "complete": bool(result["render_ready"]),
                "empty": False,
                "truncated": bool(result["truncated"]),
            },
        )

    def _render_review_animation(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            output = self.path_policy.resolve_read(arguments["output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, output, require_store=True)
            self.path_policy.resolve_write(output)
            artifact_id = safe_artifact_id(str(arguments["artifact_id"]))
            file_format = str(arguments.get("format", "mp4"))
            target = output / "review" / "mcp_animations" / f"{artifact_id}.{file_format}"
            companion_paths = (
                target,
                target.with_suffix(".contact-sheet.png"),
                target.with_suffix(".quality.json"),
            )
            if any(path.exists() for path in companion_paths):
                raise ValueError("The planned animation target already exists; choose a new artifact_id and re-plan.")
            target = self.path_policy.resolve_write(target)
            spec = animation_spec_from_mapping(arguments, source="oel_mcp_review_animation_v1")
            artifact = render_planned_review_animation(
                output,
                spec,
                animation_plan_id=str(arguments["animation_plan_id"]),
                path=target,
            )
            artifact_paths = [
                artifact.path,
                artifact.contact_sheet_path,
                artifact.quality_receipt_path,
            ]
            artifact_payload = {
                "artifact_id": artifact.artifact_id,
                "path": str(artifact.path),
                "relative_path": artifact.relative_path,
                "path_exists": artifact.path.is_file(),
                "contact_sheet_path": str(artifact.contact_sheet_path),
                "quality_receipt_path": str(artifact.quality_receipt_path),
                "row_count": artifact.row_count,
                "truncated": artifact.truncated,
                "spec": asdict(artifact.spec),
                "qa": dict(artifact.qa),
            }
            complete = bool(
                all(path.is_file() for path in artifact_paths)
                and not artifact.truncated
                and artifact.qa.get("automated_status") == "passed"
            )
            manifest = manifest_base(
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
            )
            manifest.update(
                {
                    "animation_plan_id": str(arguments["animation_plan_id"]),
                    "artifact_id": artifact.artifact_id,
                    "animation_spec": asdict(spec),
                    "qa": dict(artifact.qa),
                }
            )
            manifest = complete_manifest(
                manifest,
                status="completed" if complete else "partial",
                artifacts=[str(path) for path in artifact_paths if path.is_file()],
            )
            manifest_path = write_execution_manifest(
                output / "review" / "mcp_animations",
                manifest,
                filename=f"{artifact_id}.manifest.json",
            )
            return {
                "status": "completed" if complete else "partial",
                "output_dir": str(output),
                "animation_plan_id": str(arguments["animation_plan_id"]),
                "artifact": artifact_payload,
                "manifest_path": str(manifest_path),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": False,
                "truncated": bool(result["artifact"].get("truncated")),
            },
        )

    def _run_agent_task(
        self,
        contract: ToolContract,
        arguments: dict[str, Any],
        *,
        cancel_event: Event | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            recipe_id = str(arguments["recipe_id"])
            recipe = get_recipe(recipe_id)
            if not is_public_mcp_recipe(recipe):
                raise ValueError("Only supported public scenario-run task recipes are available.")
            output = self.path_policy.resolve_write(arguments["output_dir"])
            prepared = prepare_scenario(
                config_path=Path(__file__).resolve().parents[2] / recipe.config_path,
                output_dir=output,
                resource_profile=str(arguments["resource_profile"]),
                path_policy=self.path_policy,
            )
            estimate = resource_estimate(prepared)
            require_safe_resource_estimate(estimate)
            ensure_new_output_dir(output)
            manifest = manifest_base(
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                prepared=prepared,
            )
            manifest.update(
                {"recipe_id": recipe_id, "resource_profile": arguments["resource_profile"], "output_dir": str(output)}
            )
            manifest["resource_estimate"] = estimate
            manifest_path = write_execution_manifest(output, manifest)
            try:
                if progress:
                    progress(1, 3, "Prepared the approved public task recipe.")
                packet = run_recipe(
                    recipe_id,
                    output_dir=output,
                    make_plots=bool(arguments.get("make_plots", False)),
                    style_name=str(arguments.get("style", "oel_dark")),
                    max_rows=int(arguments.get("max_rows", 50)),
                    resource_profile=str(arguments["resource_profile"]),
                    step_callback=_task_cancellation_callback(cancel_event),
                )
                status = str(packet.get("status", "failed"))
                prepared_packet = dict((packet.get("configs") or [{}])[0])
                expected_identity = {
                    "source_config_sha256": prepared.raw_sha256,
                    "normalized_config_sha256": prepared.normalized_sha256,
                    "validation_id": prepared.validation_id,
                }
                actual_identity = {key: prepared_packet.get(key) for key in expected_identity}
                if status not in {"failed", "cancelled"} and actual_identity != expected_identity:
                    raise ValueError("Executed task config does not match the preflighted scenario identity.")
                _task_cancellation_callback(cancel_event)()
                if status not in {"completed", "partial", "failed", "cancelled"}:
                    status = "failed"
                artifacts = [
                    str(item.get("resolved_path") or item.get("path"))
                    for item in packet.get("artifacts", [])
                    if item.get("resolved_path") or item.get("path")
                ]
                if packet.get("packet_path"):
                    artifacts.append(str(packet["packet_path"]))
                complete_manifest(manifest, status=status, artifacts=artifacts)
                write_execution_manifest(output, manifest)
                if progress:
                    progress(3, 3, f"Task evidence packet finished with status {status}.")
                return {
                    "status": status,
                    "recipe_id": recipe_id,
                    "recipe_maturity": recipe.maturity,
                    "output_dir": str(output),
                    "evidence_summary": dict(packet.get("evidence_summary", {}) or {}),
                    "packet_path": str(packet.get("packet_path", "")),
                    "manifest_path": str(manifest_path),
                    "artifacts": [dict(item) for item in packet.get("artifacts", [])],
                    "failure_hints": [dict(item) for item in list(packet.get("failure_hints", []) or [])],
                }
            except AgentTaskCancelled:
                complete_manifest(manifest, status="cancelled", cancelled=True, error_type="MCPExecutionCancelled")
                write_execution_manifest(output, manifest)
                return {
                    "status": "cancelled",
                    "recipe_id": recipe_id,
                    "recipe_maturity": recipe.maturity,
                    "output_dir": str(output),
                    "evidence_summary": {"ready_to_cite": False},
                    "packet_path": "",
                    "manifest_path": str(manifest_path),
                    "artifacts": [],
                    "failure_hints": [],
                }
            except Exception as exc:
                complete_manifest(manifest, status="failed", error_type=type(exc).__name__)
                write_execution_manifest(output, manifest)
                raise

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: (
                result["status"] if result["status"] in {"completed", "failed"} else "partial"
            ),
            evidence=lambda result: {"complete": result["status"] == "completed", "empty": False, "truncated": False},
        )

    def _prepare_report_packet(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["source_output_dir"], kind="directory")
            _authorize_review_inputs(self.path_policy, source, require_store=False, report_sources=True)
            output = self.path_policy.resolve_write(arguments["packet_output_dir"])
            ensure_new_output_dir(output)
            packet_id = safe_artifact_id(str(arguments["packet_id"]))
            return prepare_report_packet_artifacts(
                source_output_dir=source,
                packet_output_dir=output,
                packet_id=packet_id,
                query_names=[str(item) for item in list(arguments.get("query_names", []) or [])],
                max_rows=int(arguments.get("max_rows", 50)),
                handling=dict(arguments["handling"]),
                approval_id=str(arguments["approval"]["approval_id"]),
            )

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: result["status"],
            evidence=lambda result: {
                "complete": result["status"] == "completed",
                "empty": result["artifact_count"] == 0,
                "truncated": False,
            },
        )

    def _audit_report(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            report = self.path_policy.resolve_read(arguments["report_path"], kind="file")
            packet = self.path_policy.resolve_read(arguments["packet_path"], kind="file")
            preparation_manifest = self.path_policy.resolve_read_child(
                packet.parent,
                "mcp_execution_manifest.json",
                kind="file",
                required=False,
            )
            require_file_size(
                packet,
                maximum=int(arguments.get("max_packet_bytes", MAX_REPORT_SOURCE_BYTES)),
            )
            packet_payload = json.loads(packet.read_text(encoding="utf-8"))
            if not isinstance(packet_payload, dict):
                raise ValueError("The report packet JSON root must be an object.")
            source = self.path_policy.resolve_read(
                str(packet_payload.get("source_output_dir", "")), kind="directory"
            )
            _authorize_review_inputs(self.path_policy, source, require_store=False, report_sources=True)
            output = self.path_policy.resolve_write(arguments["audit_output_dir"])
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("The approved output directory must be new or empty.")
            return audit_report_artifacts(
                report_path=report,
                packet_path=packet,
                audit_output_dir=output,
                author=str(arguments["author"]),
                model=str(arguments.get("model", "")),
                approval_id=str(arguments["approval"]["approval_id"]),
                preparation_manifest_path=(
                    preparation_manifest if preparation_manifest.is_file() else None
                ),
            )

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "passed" else "partial",
            evidence=lambda result: {
                "complete": result["status"] == "passed",
                "empty": False,
                "truncated": False,
            },
        )

    def _inspect_handoff(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["path"], kind="file")
            require_file_size(source, maximum=MAX_MANIFEST_BYTES)
            _authorize_interchange_sources(self.path_policy, source)
            raw = inspect_path(source, verify_sources=bool(arguments.get("verify_sources", True)))
            validation = dict(raw.get("validation", {}) or {})
            product_kind = str(raw.get("product_kind", "") or "")
            return {
                "document_type": str(raw.get("document_type", "unknown")),
                "schema_id": str(raw.get("schema_id", "")),
                "schema_version": raw.get("schema_version"),
                "identifier": str(raw.get("product_id") or raw.get("manifest_id") or ""),
                "product_kind": product_kind,
                "quality": dict(raw.get("quality", {}) or {}),
                "freshness": dict(raw.get("freshness", {}) or {}),
                "validation": validation,
                "supported_next_actions": _handoff_next_actions(
                    document_type=str(raw.get("document_type", "unknown")),
                    product_kind=product_kind,
                    promotable=bool(validation.get("promotable", False)),
                ),
                "source_path": str(source),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if bool(result["validation"].get("valid")) else "partial",
            evidence=lambda result: {
                "complete": bool(result["validation"].get("valid")),
                "empty": False,
                "truncated": False,
            },
        )

    def _export_run_product(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            completed_run = self.path_policy.resolve_read(arguments["completed_run"], kind="directory")
            _authorize_completed_run_export(self.path_policy, completed_run)
            target = _new_write_file(self.path_policy, arguments["output_path"])
            product_kind = str(arguments["product_kind"])
            selector = str(arguments.get("selector", "final"))
            _validate_selector(arguments, selector=selector)
            if product_kind == "completed_run_state":
                if arguments.get("object_ids"):
                    raise ValueError("completed_run_state accepts object_id, not object_ids.")
                raw = export_completed_run_state(
                    completed_run,
                    output_path=target,
                    object_id=arguments.get("object_id"),
                    selector=selector,
                    sample_index=arguments.get("sample_index"),
                    time_s=arguments.get("time_s"),
                    event_id=arguments.get("event_id"),
                    epoch_jd_utc=arguments.get("epoch_jd_utc"),
                    overwrite=False,
                )
                object_ids = [str(raw.get("object_id", ""))]
                selection = dict(raw.get("selection", {}) or {})
                event_id = str(arguments.get("event_id", "") or "")
            elif product_kind == "completed_run_snapshot":
                object_ids = [str(item) for item in list(arguments.get("object_ids", []) or [])]
                if len(object_ids) < 2:
                    raise ValueError("completed_run_snapshot requires at least two object_ids.")
                raw = export_completed_run_snapshot(
                    completed_run,
                    output_path=target,
                    object_ids=object_ids,
                    selector=selector,
                    sample_index=arguments.get("sample_index"),
                    time_s=arguments.get("time_s"),
                    event_id=arguments.get("event_id"),
                    epoch_jd_utc=arguments.get("epoch_jd_utc"),
                    overwrite=False,
                )
                selection = {
                    "selector": selector,
                    "sample_index": raw.get("sample_index"),
                    "time_s": raw.get("time_s"),
                }
                event_id = str(arguments.get("event_id", "") or "")
            else:
                if selector != "final" or any(arguments.get(key) is not None for key in ("sample_index", "time_s", "epoch_jd_utc")):
                    raise ValueError("maneuver_detection supports event_id/observer_id/target_id selection only.")
                raw = export_maneuver_detection_product(
                    completed_run,
                    output_path=target,
                    event_id=arguments.get("event_id"),
                    observer_id=arguments.get("observer_id"),
                    target_id=arguments.get("target_id"),
                    overwrite=False,
                )
                object_ids = [str(raw.get("observer_id", "")), str(raw.get("target_id", ""))]
                selection = {"event_id": raw.get("event_id")}
                event_id = str(raw.get("event_id", "") or "")
            manifest_path = _write_operation_manifest(
                path_policy=self.path_policy,
                target=target,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=[str(target)],
                extra={"product_kind": product_kind, "product_id": str(raw.get("product_id", ""))},
            )
            return {
                "status": "completed",
                "product_kind": product_kind,
                "product_path": str(target),
                "product_id": str(raw.get("product_id", "")),
                "selection": selection,
                "object_ids": object_ids,
                "event_id": event_id,
                "operation_manifest_path": str(manifest_path),
                "execution_occurred": False,
            }

        return self._envelope(contract=contract, arguments=arguments, operation=operation)

    def _emit_scenario_overlay(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            source = self.path_policy.resolve_read(arguments["source_scenario"], kind="file")
            overlay_path = self.path_policy.resolve_read(arguments["overlay_path"], kind="file")
            require_file_size(source, maximum=2_000_000)
            require_file_size(overlay_path, maximum=MAX_MANIFEST_BYTES)
            target = _new_write_file(self.path_policy, arguments["output_path"])
            raw = emit_scenario_overlay(
                source,
                load_scenario_overlay(overlay_path),
                overlay_id=str(arguments["overlay_id"]),
                output_path=target,
                rationale=str(arguments["rationale"]),
            )
            product = load_interchange_document(target)
            operations = list(dict(dict(product.get("payload", {}) or {}).get("patch", {}) or {}).get("operations", []) or [])
            manifest_path = _write_operation_manifest(
                path_policy=self.path_policy,
                target=target,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=[str(target)],
                extra={"product_id": str(raw.get("product_id", "")), "overlay_id": str(arguments["overlay_id"])},
            )
            return {
                "status": "completed",
                "product_path": str(target),
                "product_id": str(raw.get("product_id", "")),
                "overlay_id": str(arguments["overlay_id"]),
                "operation_count": len(operations),
                "operation_manifest_path": str(manifest_path),
                "execution_occurred": False,
            }

        return self._envelope(contract=contract, arguments=arguments, operation=operation)

    def _materialize_onp_handoff(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            product_path = self.path_policy.resolve_read(arguments["product_path"], kind="file")
            require_file_size(product_path, maximum=MAX_MANIFEST_BYTES)
            _authorize_interchange_sources(self.path_policy, product_path)
            scenario_path = _new_write_file(self.path_policy, arguments["scenario_path"])
            run_output_dir = self.path_policy.resolve_write(arguments["run_output_dir"])
            document = load_interchange_document(product_path)
            product_kind = str(document.get("product_kind", ""))
            common = {
                "scenario_name": str(arguments["scenario_name"]),
                "scenario_path": scenario_path,
                "output_dir": run_output_dir,
                "duration_s": float(arguments["duration_s"]),
                "dt_s": float(arguments["dt_s"]),
                "trust_plugins": bool(arguments.get("trust_plugins", False)),
                "overwrite": False,
            }
            if product_kind == "oel.completed_run_snapshot":
                raw = materialize_snapshot_onp(product_path, **common)
                status = "materialized" if raw.get("status") == "materialized" else "blocked"
                source_product_id = str(raw.get("source_product_id", ""))
                handoff_manifest_path = ""
                handoff_manifest_id = ""
                validation = {
                    "safe_validation": dict(raw.get("safe_validation", {}) or {}),
                    "ordinary_validation": dict(raw.get("ordinary_validation", {}) or {}),
                }
                failures = [] if status == "materialized" else [{"code": "snapshot.materialization_failed"}]
                next_action = "Validate the generated scenario before separately authorized execution."
            elif product_kind in {"oel.state_estimate", "oel.completed_run_state"}:
                raw = materialize_onp(product_path, **common)
                status = str(raw.get("status", "blocked"))
                source_product_id = str(document.get("product_id", ""))
                handoff_manifest_path = str(raw.get("manifest_path", ""))
                handoff_manifest_id = str(raw.get("manifest_id", ""))
                validation = {
                    "product": dict(raw.get("product_validation", {}) or {}),
                    "safe_validation": dict(raw.get("safe_validation", {}) or {}),
                    "ordinary_validation": dict(raw.get("ordinary_validation", {}) or {}),
                }
                failures = [dict(item) for item in list(raw.get("failures", []) or [])]
                next_action = str(raw.get("recommended_next_action", ""))
            else:
                raise ValueError("ONP handoff materialization supports accepted absolute-state and atomic-snapshot products only.")
            artifacts = [str(path) for path in (scenario_path, Path(handoff_manifest_path) if handoff_manifest_path else None) if path and Path(path).is_file()]
            operation_manifest = _write_operation_manifest(
                path_policy=self.path_policy,
                target=scenario_path,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=artifacts,
                status="completed" if status == "materialized" else "partial",
                extra={"source_product_id": source_product_id, "handoff_manifest_id": handoff_manifest_id},
            )
            return {
                "status": status,
                "scenario_path": str(scenario_path),
                "manifest_path": handoff_manifest_path,
                "manifest_id": handoff_manifest_id,
                "source_product_id": source_product_id,
                "validation": validation,
                "failures": failures,
                "recommended_next_action": next_action,
                "operation_manifest_path": str(operation_manifest),
                "execution_occurred": False,
                "execution_authorized": False,
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "materialized" else "partial",
            evidence=lambda result: {"complete": result["status"] == "materialized", "empty": False, "truncated": False},
        )

    def _materialize_scenario_patch(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            patch_product = self.path_policy.resolve_read(arguments["patch_product"], kind="file")
            source_scenario = self.path_policy.resolve_read(arguments["source_scenario"], kind="file")
            require_file_size(patch_product, maximum=MAX_MANIFEST_BYTES)
            _authorize_interchange_sources(self.path_policy, patch_product)
            scenario_path = _new_write_file(self.path_policy, arguments["scenario_path"])
            run_output_dir = self.path_policy.resolve_write(arguments["run_output_dir"])
            raw = materialize_scenario_patch(
                patch_product,
                source_scenario,
                scenario_name=str(arguments["scenario_name"]),
                scenario_path=scenario_path,
                output_dir=run_output_dir,
                trust_plugins=bool(arguments.get("trust_plugins", False)),
                overwrite=False,
            )
            product = load_interchange_document(patch_product)
            status = str(raw.get("status", "blocked"))
            artifacts = [str(path) for path in (scenario_path, Path(str(raw.get("manifest_path", "")))) if path.is_file()]
            operation_manifest = _write_operation_manifest(
                path_policy=self.path_policy,
                target=scenario_path,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=artifacts,
                status="completed" if status == "materialized" else "partial",
                extra={"source_product_id": str(product.get("product_id", "")), "handoff_manifest_id": str(raw.get("manifest_id", ""))},
            )
            return {
                "status": status,
                "scenario_path": str(scenario_path),
                "manifest_path": str(raw.get("manifest_path", "")),
                "manifest_id": str(raw.get("manifest_id", "")),
                "source_product_id": str(product.get("product_id", "")),
                "validation": {
                    "product": dict(raw.get("product_validation", {}) or {}),
                    "safe_validation": dict(raw.get("safe_validation", {}) or {}),
                    "ordinary_validation": dict(raw.get("ordinary_validation", {}) or {}),
                },
                "failures": [dict(item) for item in list(raw.get("failures", []) or [])],
                "recommended_next_action": str(raw.get("recommended_next_action", "")),
                "operation_manifest_path": str(operation_manifest),
                "execution_occurred": False,
                "execution_authorized": False,
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "materialized" else "partial",
            evidence=lambda result: {"complete": result["status"] == "materialized", "empty": False, "truncated": False},
        )

    def _compare_handoff(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            product = self.path_policy.resolve_read(arguments["product_path"], kind="file")
            scenario = self.path_policy.resolve_read(arguments["scenario_path"], kind="file")
            manifest = (
                self.path_policy.resolve_read(arguments["manifest_path"], kind="file")
                if arguments.get("manifest_path")
                else self.path_policy.resolve_read(
                    scenario.with_name(f"{scenario.stem}.handoff_manifest.json"), kind="file"
                )
            )
            run_output = (
                self.path_policy.resolve_read(arguments["run_output_dir"], kind="directory")
                if arguments.get("run_output_dir")
                else None
            )
            if run_output is not None:
                _authorize_review_inputs(self.path_policy, run_output, require_store=True)
            output = _new_write_file(self.path_policy, arguments["output_path"])
            raw = compare_handoff(
                product,
                scenario,
                manifest_path=manifest,
                run_output_dir=run_output,
                output_path=output,
            )
            operation_manifest = _write_operation_manifest(
                path_policy=self.path_policy,
                target=output,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=[str(output)],
                status="completed" if raw.get("status") == "equivalent" else "partial",
                extra={"comparison_id": str(raw.get("comparison_id", ""))},
            )
            return {
                "status": str(raw.get("status", "failed")),
                "comparison_id": str(raw.get("comparison_id", "")),
                "output_path": str(output),
                "source": dict(raw.get("source", {}) or {}),
                "materialization": dict(raw.get("materialization", {}) or {}),
                "summary": dict(raw.get("summary", {}) or {}),
                "execution_evidence": dict(raw.get("execution_evidence", {}) or {}),
                "non_claims": [str(item) for item in list(raw.get("non_claims", []) or [])],
                "operation_manifest_path": str(operation_manifest),
            }

        return self._envelope(
            contract=contract,
            arguments=arguments,
            operation=operation,
            outcome_status=lambda result: "completed" if result["status"] == "equivalent" else "partial",
            evidence=lambda result: {"complete": result["status"] == "equivalent", "empty": False, "truncated": False},
        )

    def _assess_maneuver_readiness(self, contract: ToolContract, arguments: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            completed_run = self.path_policy.resolve_read(arguments["completed_run"], kind="directory")
            _authorize_completed_run_export(self.path_policy, completed_run)
            output = _new_write_file(self.path_policy, arguments["output_path"])
            raw = build_maneuver_readiness_packet(
                completed_run,
                object_id=str(arguments["object_id"]),
                chief_id=str(arguments["chief_id"]),
                thresholds=dict(arguments["thresholds"]),
                output_path=output,
            )
            operation_manifest = _write_operation_manifest(
                path_policy=self.path_policy,
                target=output,
                tool_id=contract.tool_id,
                approval_id=str(arguments["approval"]["approval_id"]),
                artifacts=[str(output)],
                extra={"verdict": str(raw.get("verdict", "unknown"))},
            )
            return {
                "status": "completed",
                "verdict": str(raw.get("verdict", "unknown")),
                "object_id": str(raw.get("object_id", "")),
                "chief_id": str(raw.get("chief_id", "")),
                "thresholds": dict(raw.get("thresholds", {}) or {}),
                "metrics": dict(raw.get("metrics", {}) or {}),
                "gates": [dict(item) for item in list(raw.get("gates", []) or [])],
                "packet_path": str(output),
                "operation_manifest_path": str(operation_manifest),
                "non_claims": [str(item) for item in list(raw.get("non_claims", []) or [])],
            }

        return self._envelope(contract=contract, arguments=arguments, operation=operation)

def _new_write_file(path_policy: Any, value: str | Path) -> Path:
    target = path_policy.resolve_write(value)
    if target.exists():
        raise FileExistsError("The approved output file must be new.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authorize_completed_run_export(path_policy: Any, output_dir: Path) -> None:
    _authorize_review_inputs(path_policy, output_dir, require_store=True, report_sources=True)


def _authorize_interchange_sources(path_policy: Any, source: Path) -> None:
    document = load_interchange_document(source)
    provenance = dict(document.get("provenance", {}) or {})
    for artifact in list(provenance.get("source_artifacts", []) or []):
        row = dict(artifact or {})
        raw_path = str(row.get("path", "") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        path_policy.resolve_read(candidate, kind="file")


def _validate_selector(arguments: dict[str, Any], *, selector: str) -> None:
    required = {
        "sample_index": "sample_index",
        "time_s": "time_s",
        "event": "event_id",
    }
    key = required.get(selector)
    if key and arguments.get(key) is None:
        raise ValueError(f"selector {selector!r} requires {key}.")
    incompatible = {
        "final": ("sample_index", "time_s", "event_id"),
        "sample_index": ("time_s", "event_id"),
        "time_s": ("sample_index", "event_id"),
        "event": ("sample_index", "time_s"),
    }
    if any(arguments.get(item) is not None for item in incompatible.get(selector, ())):
        raise ValueError("Completed-run selector arguments conflict.")


def _handoff_next_actions(*, document_type: str, product_kind: str, promotable: bool) -> list[str]:
    if document_type == "manifest":
        return ["compare_handoff"]
    if document_type != "product" or not promotable:
        return ["resolve_validation_or_promotion_blockers"]
    if product_kind in {"oel.state_estimate", "oel.completed_run_state", "oel.completed_run_snapshot"}:
        return ["materialize_onp_handoff"]
    if product_kind == "oel.scenario_patch":
        return ["materialize_scenario_patch"]
    if product_kind == "oel.maneuver_detection":
        return ["prepare_measured_event_evidence"]
    if product_kind == "oel.ogp_mean_element_product":
        return ["materialize_ogp_through_supported_cli"]
    return ["inspect_only"]


def _write_operation_manifest(
    *,
    path_policy: Any,
    target: Path,
    tool_id: str,
    approval_id: str,
    artifacts: list[str],
    status: str = "completed",
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest_target = path_policy.resolve_write(
        target.with_name(f"{target.stem}.mcp_operation_manifest.json")
    )
    if manifest_target.exists():
        raise FileExistsError("The MCP operation manifest output must be new.")
    manifest = manifest_base(tool_id=tool_id, approval_id=approval_id)
    manifest.update(dict(extra or {}))
    complete_manifest(manifest, status=status, artifacts=artifacts)
    return write_execution_manifest(manifest_target.parent, manifest, filename=manifest_target.name)


def _run_projection(payload: dict[str, Any]) -> dict[str, Any]:
    run = dict(payload.get("run", {}) or payload.get("summary", {}) or {})
    keys = (
        "scenario_name",
        "duration_s",
        "dt_s",
        "samples",
        "termination_status",
        "output_dir",
        "closest_approach_km",
        "final_range_km",
        "total_delta_v_m_s",
    )
    return {key: run[key] for key in keys if key in run}


def _existing_artifacts(output_dir: Path, materialized: Path) -> list[str]:
    candidates = (
        materialized,
        output_dir / "index.md",
        output_dir / "master_run_summary.json",
        output_dir / "review" / "run.sqlite",
        output_dir / "agent_evidence_packet.json",
    )
    artifacts = [path for path in candidates if path.exists()]
    inventory_path = output_dir / ".oel_run_artifacts.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
        inventory = {}
    root = output_dir.resolve()
    for row in list(dict(inventory or {}).get("paths", []) or []):
        relative = Path(str(dict(row or {}).get("path", "") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            continue
        candidate = root / relative
        if candidate.is_file() and not candidate.is_symlink():
            artifacts.append(candidate)
    return [str(path) for path in dict.fromkeys(artifacts)]


def _require_review_store_budget(db_path: Path) -> None:
    sidecars = (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm"))
    total = 0
    for path in sidecars:
        if not path.is_file():
            continue
        require_file_size(path, maximum=MAX_REVIEW_STORE_BYTES)
        total += path.stat().st_size
        if total > MAX_REVIEW_STORE_BYTES:
            raise ValueError("Review store and SQLite sidecars exceed the file-size budget.")


def _authorize_review_inputs(
    path_policy: Any,
    output_dir: Path,
    *,
    require_store: bool,
    report_sources: bool = False,
) -> None:
    logical_store = output_dir / "review" / "run.sqlite"
    capsule_store = output_dir / "review" / "run.sqlite.gz"
    if require_store and not logical_store.is_file() and not capsule_store.is_file():
        raise FileNotFoundError(
            f"Required review evidence is missing: expected {logical_store} or {capsule_store}."
        )
    children = [
        (("review", "run.sqlite"), "file", False),
        (("review", "run.sqlite.gz"), "file", False),
        (("review", "evidence_capsule.json"), "file", False),
        (("review", "schema.json"), "file", False),
        (("review", "saved_views.json"), "file", False),
        (("review", "workflow_manifest.json"), "file", False),
        (("mcp_execution_manifest.json",), "file", False),
    ]
    if report_sources:
        children.extend(
            [
                (("master_run_summary.json",), "file", False),
                (("agent_evidence_packet.json",), "file", False),
                (("index.md",), "file", False),
            ]
        )
    for parts, kind, required in children:
        resolved = path_policy.resolve_read_child(output_dir, *parts, kind=kind, required=required)
        if resolved.is_file():
            is_review_store = resolved.name in {"run.sqlite", "run.sqlite.gz"}
            require_file_size(
                resolved,
                maximum=MAX_REVIEW_STORE_BYTES if is_review_store else MAX_MANIFEST_BYTES,
            )


def _authorize_study_bundle(path_policy: Any, bundle: Path) -> None:
    entries = list(bundle.rglob("*"))
    if len(entries) > 64:
        raise ValueError("Study bundle exceeds the bounded 64-entry inspection limit.")
    total_bytes = 0
    for entry in entries:
        relative = entry.relative_to(bundle)
        resolved = path_policy.resolve_read_child(
            bundle, *relative.parts, kind="directory" if entry.is_dir() else "file"
        )
        if resolved.is_symlink():
            raise ValueError("Study bundle entries must not be symlinks.")
        if resolved.is_file():
            require_file_size(resolved, maximum=16 * 1024 * 1024)
            total_bytes += resolved.stat().st_size
    if total_bytes > 64 * 1024 * 1024:
        raise ValueError("Study bundle exceeds the bounded 64 MiB inspection limit.")


def _require_argument_fields(arguments: dict[str, Any], operation: str, *names: str) -> None:
    missing = [name for name in names if arguments.get(name) is None]
    if missing:
        raise ValueError(f"{operation} requires fields: {', '.join(missing)}.")


def _parse_utc_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    text = str(value)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _frame_time_projection(
    operation: str,
    *,
    result: dict[str, Any],
    receipt: dict[str, Any] | None,
    eop_source: dict[str, Any] | None = None,
    status: str = "converted",
) -> dict[str, Any]:
    return {
        "status": status,
        "operation": operation,
        "result": result,
        "receipt": receipt,
        "eop_source": eop_source,
        "execution_occurred": False,
        "non_claims": [
            "Frame and time conversion does not validate the physical accuracy of upstream state data.",
            "Earth-orientation freshness and provenance remain part of the returned evidence contract.",
        ],
    }


def _incomplete_manifest_status(output_dir: Path) -> dict[str, Any] | None:
    provenance = _execution_provenance(output_dir)
    if not provenance["available"]:
        return None
    if provenance["status"] == "completed" and provenance["artifacts_complete"]:
        return None
    if provenance["status"] == "completed":
        return {
            "code": "mcp_execution_artifact_integrity_failed",
            "status": "completed",
            "next_step": "Restore the manifest-listed artifacts or rerun the source workflow.",
        }
    return {
        "code": "mcp_execution_not_completed",
        "status": provenance["status"],
        "next_step": "Use a completed execution or rerun the source workflow.",
    }


def _manifest_artifacts_valid(manifest: dict[str, Any], output: Path) -> bool:
    if not bool(manifest.get("artifacts_complete", False)):
        return False
    # Review plotting updates the human-readable index with generated-artifact
    # links after the source run settles. It is derived navigation metadata, so
    # it cannot be part of the immutable source-artifact integrity set.
    mutable_paths = {(output / "index.md").resolve()}
    records = [
        dict(raw or {})
        for raw in list(manifest.get("artifact_records", []) or [])
        if Path(str(dict(raw or {}).get("path", "") or "")).resolve() not in mutable_paths
    ]
    artifacts = [
        str(path)
        for path in list(manifest.get("artifacts", []) or [])
        if Path(str(path)).resolve() not in mutable_paths
    ]
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


def _execution_provenance(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "mcp_execution_manifest.json"
    empty = {
        "available": False,
        "status": "",
        "tool_id": "",
        "validation_id": "",
        "source_config_sha256": "",
        "normalized_config_sha256": "",
        "resource_profile": "",
        "artifacts_complete": False,
        "artifact_count": 0,
        "cancelled": False,
        "started_utc": None,
        "completed_utc": None,
        "artifact_records_valid": False,
    }
    if not manifest_path.is_file():
        return empty
    require_file_size(manifest_path, maximum=MAX_MANIFEST_BYTES)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The MCP execution manifest must contain a JSON object.")
    artifact_records_valid = _manifest_artifacts_valid(payload, output_dir)
    return {
        "available": True,
        "status": str(payload.get("status", "")),
        "tool_id": str(payload.get("tool_id", "")),
        "validation_id": str(payload.get("validation_id", "")),
        "source_config_sha256": str(payload.get("source_config_sha256", "")),
        "normalized_config_sha256": str(payload.get("normalized_config_sha256", "")),
        "resource_profile": str(payload.get("resource_profile", "")),
        "artifacts_complete": bool(payload.get("artifacts_complete", False)) and artifact_records_valid,
        "artifact_count": len(list(payload.get("artifacts", []) or [])),
        "cancelled": bool(payload.get("cancelled", False)),
        "started_utc": payload.get("started_utc"),
        "completed_utc": payload.get("completed_utc"),
        "artifact_records_valid": artifact_records_valid,
    }


def _validation_next_step(
    *,
    validation: dict[str, Any],
    resource_action: str,
    trust_plugins: bool,
) -> str:
    if resource_action == "refuse":
        return "Reduce the resource envelope and validate again before requesting execution approval."
    if bool(validation.get("execution_ready")):
        return "Obtain an operator-configured execution approval and call run_scenario with this validation id."
    if not trust_plugins and bool(dict(validation.get("safe_validation", {}) or {}).get("ok")):
        return "Review the referenced plugins and paths, then repeat validation with an operator-configured trust approval."
    return "Resolve the reported validation issues and validate again before requesting execution approval."


def _task_cancellation_callback(cancel_event: Event | None) -> Any:
    def check(*_args: Any, **_kwargs: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AgentTaskCancelled("The MCP task was cancelled at a deterministic workflow boundary.")

    return check
