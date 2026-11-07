from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from sim.agent_task.failures import diagnose_failure
from sim.agent_task.models import AgentPlotRecipe, AgentTaskRecipe, EvidencePacket
from sim.agent_task.plot_recipes import get_plot_recipe, review_plot_spec
from sim.agent_task.recipes import get_recipe
from sim.agent_task.semantics import get_semantic_metric, semantic_metric_dicts, semantic_metric_request_rows
from sim.api import SimulationWorkspace
from sim.config import scenario_config_from_dict
from sim.execution import run_simulation_config_file
from sim.execution.service import SimulationExecutionService
from sim.resource_limits import apply_resource_profile_to_config_dict
from sim.review import (
    ReviewQueryError,
    ReviewStoreNotFoundError,
    ReviewWorkspace,
    get_saved_review_query,
    load_workflow_manifest,
)
from sim.review.plotting import ReviewPlotArtifact, save_review_plot
from sim.security import ConfigPathPolicy

if TYPE_CHECKING:
    from sim.security.output_writes import OutputWriter

DEFAULT_INSPECTION_QUERIES = ("run_metadata", "objects", "artifacts")
DEFAULT_COMPARISON_METRICS = (
    "initial_range_km",
    "final_range_km",
    "final_range_rate_km_s",
    "closest_approach_km",
    "closest_approach_time_s",
    "total_delta_v_m_s",
)

_PREPARED_CONFIG_CACHE: dict[str, tuple[str, Any]] = {}
_EXPECTED_PREPARED_CONFIG_DIGESTS: dict[str, str] = {}
_AGENT_TASK_EXECUTION_SERVICE = SimulationExecutionService()
_RUN_SIMULATION_CONFIG_FILE_ORIGINAL = run_simulation_config_file


def _lexical_path(value: str | Path) -> Path:
    """Return an absolute path without traversing producer-controlled symlinks.

    The MCP writer owns the physical descriptor and deliberately keeps a
    logical output path that may be renamed while a producer is running.  A
    normal ``Path.resolve()`` here would follow a replacement symlink and
    change the identity reported by the task packet.  ``abspath`` only
    normalizes the lexical components; the descriptor writer performs the
    no-follow checks when it opens or publishes bytes.
    """

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(os.fspath(path)))


class AgentTaskCancelled(RuntimeError):
    """Raised by a caller callback to stop a task at a deterministic step boundary."""


class _OutputBinding:
    """Bind logical producer paths to a descriptor-backed physical writer."""

    def __init__(self, writer: Any, logical_root: str | Path) -> None:
        self.writer = writer
        self.logical_root = _lexical_path(logical_root)
        self.writer_logical_root = _lexical_path(writer.logical_root)
        self.physical_root = _lexical_path(writer.path)
        try:
            self.logical_root.relative_to(self.writer_logical_root)
        except ValueError as exc:
            raise ValueError(
                "Agent-task output logical root must be within the bound writer logical root."
            ) from exc

    def _relative(self, logical_path: str | Path) -> Path:
        path = _lexical_path(logical_path)
        try:
            return path.relative_to(self.writer_logical_root)
        except ValueError as exc:
            raise ValueError(
                f"Agent-task output path is outside the bound writer logical root: {path}"
            ) from exc

    def physical(self, logical_path: str | Path) -> Path:
        return self.physical_root / self._relative(logical_path)

    def ensure_directory(self, logical_path: str | Path) -> Path:
        path = _lexical_path(logical_path)
        if path == self.logical_root and path == self.writer_logical_root:
            if not self.physical_root.is_dir():
                raise ValueError("The bound agent-task output writer root is not a directory.")
            return self.physical_root
        from sim.security.output_writes import bind_output_writer, mkdir_output

        with bind_output_writer(self.writer):
            return mkdir_output(path)

    def write_bytes(self, logical_path: str | Path, payload: bytes) -> Path:
        from sim.security.output_writes import (
            SecureWriteConflict,
            bind_output_writer,
            write_output_bytes,
        )

        path = _lexical_path(logical_path)
        relative = self._relative(path)
        with bind_output_writer(self.writer):
            try:
                write_output_bytes(path, payload)
            except SecureWriteConflict:
                self.writer.update_existing_bytes(relative, payload)
        return path

    def logicalize(self, value: Any) -> Any:
        from sim.security.output_writes import rewrite_staged_paths

        return rewrite_staged_paths(value, self.physical_root, self.writer_logical_root)


def _bind_output_writer(
    writer: OutputWriter | None, logical_root: str | Path
) -> _OutputBinding | None:
    if writer is None:
        return None
    return _OutputBinding(writer, logical_root)


@contextlib.contextmanager
def _bound_output_writer(writer: OutputWriter | None):
    if writer is None:
        yield
        return
    from sim.security.output_writes import bind_output_writer

    with bind_output_writer(writer):
        yield


def _artifact_status(
    artifacts: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    binding: _OutputBinding | None,
) -> list[dict[str, Any]]:
    logical_root = _lexical_path(output_dir)
    physical_root = binding.physical(logical_root) if binding is not None else logical_root
    physical_artifacts: list[dict[str, Any]] = []
    for item in artifacts:
        row = dict(item)
        raw_path = str(row.get("path") or "")
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = logical_root / candidate
            if binding is not None:
                try:
                    candidate.relative_to(binding.physical_root)
                except ValueError:
                    candidate = binding.physical(candidate)
            row["path"] = str(candidate)
        physical_artifacts.append(row)
    checked = _artifacts_with_path_status(physical_artifacts, output_dir=physical_root)
    return binding.logicalize(checked) if binding is not None else checked


def run_recipe(
    recipe_id: str,
    *,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    make_plots: bool = False,
    style_name: str = "oel_dark",
    max_rows: int = 50,
    resource_profile: str = "config",
    step_callback: Any | None = None,
    output_writer: OutputWriter | None = None,
    protected_review: bool | None = None,
) -> dict[str, Any]:
    recipe = get_recipe(recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown agent task recipe: {recipe_id}")
    if output_writer is not None and output_dir is None and output_root is None:
        output_dir = _lexical_path(output_writer.logical_root)
    prepared = prepare_recipe_config(
        recipe,
        output_dir=output_dir,
        output_root=output_root,
        resource_profile=resource_profile,
        output_writer=output_writer,
    )
    task_id = recipe.recipe_id
    binding = _bind_output_writer(output_writer, prepared["output_dir"])
    execution_config_path = (
        str(binding.physical(prepared["config_path"]))
        if binding is not None
        else prepared["config_path"]
    )
    validation_workspace = _workspace()
    if binding is not None:
        validation_workspace = SimulationWorkspace(
            workspace_root=_repo_root(),
            read_roots=(Path(prepared["source_config"]).parent,),
            write_roots=(Path(prepared["output_dir"]),),
            allow_config_dir_writes=False,
        )
    validation = validation_workspace.validate(execution_config_path)
    if binding is not None:
        validation = binding.logicalize(validation)
    packet = EvidencePacket(
        task_id=task_id,
        status="validated" if bool(validation.get("ok")) else "failed",
        generated_utc=_utc_now(),
        recipe=recipe.to_dict(),
        configs=[prepared],
        validation=validation,
        semantic_metric_requests=semantic_metric_request_rows(recipe.semantic_metric_names),
        semantic_metrics=semantic_metric_dicts(recipe.semantic_metric_names),
    )
    if not bool(validation.get("ok")):
        packet.failure_hints = [
            hint.to_dict() for hint in diagnose_failure(validation=validation, output_dir=prepared["output_dir"])
        ]
        _discard_prepared_config(execution_config_path)
        return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)
    if dry_run:
        packet.caveats.append("Dry run requested: scenario was validated but not executed.")
        _discard_prepared_config(execution_config_path)
        return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)

    try:
        with _bound_output_writer(output_writer):
            payload = _run_prepared_config(execution_config_path, step_callback=step_callback)
        packet.run = binding.logicalize(_run_summary(payload)) if binding is not None else _run_summary(payload)
        packet.status = "completed"
    except AgentTaskCancelled as exc:
        packet.status = "cancelled"
        packet.run = {"cancelled": True, "error": str(exc)}
        partial_root = binding.physical(prepared["output_dir"]) if binding is not None else prepared["output_dir"]
        packet.artifacts = _partial_output_artifacts(partial_root)
        if binding is not None:
            packet.artifacts = binding.logicalize(packet.artifacts)
        packet.artifact_summary = _summarize_artifacts(packet.artifacts)
        packet.caveats.append("Execution was cancelled at a deterministic workflow boundary; artifacts are partial.")
        return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)
    except Exception as exc:
        packet.status = "failed"
        packet.run = {"error": str(exc)}
        packet.failure_hints = [
            hint.to_dict()
            for hint in diagnose_failure(str(exc), validation=validation, output_dir=prepared["output_dir"])
        ]
        return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)

    inspect_root = binding.physical(prepared["output_dir"]) if binding is not None else prepared["output_dir"]
    inspection = inspect_output(
        inspect_root,
        query_names=recipe.query_names,
        max_rows=max_rows,
        semantic_metric_names=recipe.semantic_metric_names,
        write_packet=False,
        protected=protected_review,
    )
    if binding is not None:
        inspection = binding.logicalize(inspection)
    packet.review = dict(inspection.get("review", {}) or {})
    packet.artifacts = list(inspection.get("artifacts", []) or [])
    packet.artifact_summary = dict(inspection.get("artifact_summary", {}) or _summarize_artifacts(packet.artifacts))
    packet.failure_hints.extend(list(inspection.get("failure_hints", []) or []))
    if str(inspection.get("status") or "") != "completed":
        packet.status = "partial"
    if make_plots:
        plot_workspace_dir = binding.physical(prepared["output_dir"]) if binding is not None else prepared["output_dir"]
        with _bound_output_writer(output_writer):
            packet.plots = _make_plots_for_recipe(
                plot_workspace_dir,
                recipe.plot_recipe_ids,
                style_name=style_name,
                logical_output_dir=prepared["output_dir"] if binding is not None else None,
                protected=protected_review,
            )
        if binding is not None:
            # Plot renderers return logical paths after their private
            # per-file publication.  Recheck existence through the retained
            # staging descriptor so a not-yet-published final path does not
            # turn a successful render into a false partial result.
            packet.plots = _artifact_status(
                binding.logicalize(packet.plots),
                output_dir=prepared["output_dir"],
                binding=binding,
            )
        packet.plot_summary = _summarize_plots(packet.plots)
        failed_plots = [
            dict(item)
            for item in packet.plots
            if str(item.get("status", "")).strip().lower() != "ok"
            or item.get("path_exists") is False
            or bool(item.get("truncated"))
        ]
        if failed_plots:
            packet.status = "partial"
            packet.failure_hints.append(
                {
                    "code": "review_plots_incomplete",
                    "plot_recipe_ids": [
                        str(item.get("recipe_id") or item.get("artifact_id") or "unknown")
                        for item in failed_plots
                    ],
                    "next_step": (
                        "Inspect each failed plot record and regenerate only after its renderer, "
                        "query, and source evidence are complete."
                    ),
                }
            )
    if step_callback is not None:
        try:
            step_callback(1, 1)
        except AgentTaskCancelled as exc:
            packet.status = "cancelled"
            packet.run = {"cancelled": True, "error": str(exc)}
            packet.caveats.append(
                "Execution was cancelled at a deterministic workflow boundary; artifacts are partial."
            )
            return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)
    return _write_packet(packet, prepared["output_dir"], output_writer=output_writer)


def inspect_output(
    output_dir: str | Path,
    *,
    query_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
    semantic_metric_names: tuple[str, ...] | list[str] = (),
    write_packet: bool = True,
    max_value_bytes: int | None = None,
    max_result_bytes: int | None = None,
    protected: bool | None = None,
) -> dict[str, Any]:
    outdir = _lexical_path(output_dir)
    requested_query_names = None if query_names is None else tuple(query_names)
    packet = EvidencePacket(
        task_id=f"inspect_{outdir.name}",
        status="completed",
        generated_utc=_utc_now(),
        task_type="inspect_output",
        configs=[{"output_dir": str(outdir)}],
        semantic_metric_requests=semantic_metric_request_rows(tuple(semantic_metric_names)),
        semantic_metrics=semantic_metric_dicts(tuple(semantic_metric_names)),
    )
    review: dict[str, Any] = {"output_dir": str(outdir), "queries": []}
    try:
        workspace = (
            ReviewWorkspace.open(outdir)
            if protected is None
            else ReviewWorkspace.open(outdir, protected=protected)
        )
        effective_query_names = (
            _default_inspection_queries(workspace)
            if requested_query_names is None
            else requested_query_names
        )
        review.update(
            {
                "db_path": str(workspace.db_path),
                "tables": workspace.tables(),
                "schema": workspace.schema(),
                "saved_views": workspace.saved_views(),
            }
        )
        review["queries"] = _run_saved_queries(
            workspace,
            effective_query_names,
            max_rows=max_rows,
            max_value_bytes=max_value_bytes,
            max_result_bytes=max_result_bytes,
        )
        review["query_summary"] = _summarize_query_rows(review["queries"])
        if not bool(review["query_summary"].get("evidence_complete", False)):
            packet.status = "partial"
            packet.failure_hints.append(
                {
                    "code": "review_queries_incomplete",
                    "query_summary": dict(review["query_summary"]),
                    "next_step": (
                        "Run `python -m sim.review --list-saved-queries`, select a known query, "
                        "and inspect schema columns before writing custom SQL."
                    ),
                }
            )
        packet.artifacts = _artifact_rows(review["queries"], output_dir=outdir)
    except (ReviewStoreNotFoundError, ReviewQueryError, ValueError) as exc:
        packet.status = "partial"
        packet.failure_hints = [hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=outdir)]
        review["error"] = str(exc)

    workflow_manifest = _maybe_workflow_manifest(outdir)
    if workflow_manifest:
        review["workflow_manifest"] = workflow_manifest
        packet.artifacts.extend(_workflow_artifacts(workflow_manifest, output_dir=outdir))
    packet.artifact_summary = _summarize_artifacts(packet.artifacts)
    packet.review = review
    if write_packet:
        return _write_packet(packet, outdir)
    return packet.to_dict()


def _default_inspection_queries(workspace: ReviewWorkspace) -> tuple[str, ...]:
    """Add domain summaries only when the completed run contains that evidence."""

    names = list(DEFAULT_INSPECTION_QUERIES)
    available = set(workspace.tables())
    for table_name, query_names in (
        (
            "coverage_summary",
            ("coverage_summary", "coverage_transition_summary"),
        ),
        (
            "link_summary",
            ("directed_link_summary", "directed_link_windows"),
        ),
    ):
        if table_name not in available:
            continue
        result = workspace.query(f'SELECT EXISTS(SELECT 1 FROM "{table_name}" LIMIT 1) AS present', max_rows=1)
        if result.rows and bool(result.rows[0].get("present")):
            names.extend(query_names)
    return tuple(names)


def create_plot(
    output_dir: str | Path,
    plot_recipe_id: str,
    *,
    style_name: str = "oel_dark",
    file_format: str = "png",
    artifact_id: str = "",
    path: str | Path | None = None,
    record: bool = True,
    logical_output_dir: str | Path | None = None,
    protected: bool | None = None,
) -> dict[str, Any]:
    recipe = get_plot_recipe(plot_recipe_id)
    if recipe is None:
        raise ValueError(f"Unknown agent plot recipe: {plot_recipe_id}")
    workspace = (
        ReviewWorkspace.open(output_dir)
        if protected is None
        else ReviewWorkspace.open(output_dir, protected=protected)
    )
    if logical_output_dir is not None:
        # Queries and generated-artifact indexes read from the private
        # staging directory, while their persisted relative paths belong to
        # the approved logical output root.  Keep the physical descriptors
        # from ``open`` and replace only the public workspace identity.
        workspace = replace(workspace, output_dir=_lexical_path(logical_output_dir))
    spec = review_plot_spec(
        recipe,
        style_name=style_name,
        file_format=file_format,
        artifact_id=artifact_id or recipe.artifact_id,
    )
    artifact = save_review_plot(workspace, spec, path=path, record=record)
    return _plot_artifact_dict(artifact, recipe=recipe)


def compare_configs(
    base_config: str | Path,
    candidate_config: str | Path,
    *,
    output_dir: str | Path,
    metric_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
) -> dict[str, Any]:
    outdir = _lexical_path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = tuple(metric_names or DEFAULT_COMPARISON_METRICS)
    configs = [
        _prepare_config(base_config, outdir / "base", label="base"),
        _prepare_config(candidate_config, outdir / "candidate", label="candidate"),
    ]
    workspace = _workspace()
    validations = {item["label"]: workspace.validate(item["config_path"]) for item in configs}
    packet = EvidencePacket(
        task_id=f"compare_{Path(base_config).stem}_vs_{Path(candidate_config).stem}",
        task_type="compare_configs",
        status="completed",
        generated_utc=_utc_now(),
        configs=configs,
        validation=validations,
        semantic_metric_requests=semantic_metric_request_rows(metrics),
        semantic_metrics=semantic_metric_dicts(metrics),
    )
    if not all(bool(report.get("ok")) for report in validations.values()):
        packet.status = "failed"
        packet.failure_hints = [
            hint.to_dict()
            for report in validations.values()
            for hint in diagnose_failure(validation=report, output_dir=report.get("output_dir"))
        ]
        for item in configs:
            _discard_prepared_config(item["config_path"])
        return _write_packet(packet, outdir)

    runs: dict[str, Any] = {}
    inspections: dict[str, Any] = {}
    comparison_query_names = _comparison_query_names(metrics)
    for item in configs:
        label = str(item["label"])
        try:
            runs[label] = _run_summary(_run_prepared_config(item["config_path"]))
            inspections[label] = inspect_output(
                item["output_dir"],
                query_names=comparison_query_names,
                max_rows=max_rows,
                semantic_metric_names=metrics,
                write_packet=False,
            )
        except Exception as exc:
            packet.status = "failed"
            runs[label] = {"error": str(exc)}
            packet.failure_hints.extend(
                hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=item["output_dir"])
            )
            return _write_packet(packet, outdir)

    metric_table = {label: _extract_metric_values(data, metrics) for label, data in inspections.items()}
    inspection_statuses = {label: str(data.get("status") or "") for label, data in inspections.items()}
    for label, inspection in inspections.items():
        if inspection_statuses.get(label) != "completed" and packet.status == "completed":
            packet.status = "partial"
        for hint in list(inspection.get("failure_hints", []) or []):
            packet.failure_hints.append({"label": label, **dict(hint)})
    packet.run = runs
    packet.review = {
        "base": inspections.get("base", {}).get("review", {}),
        "candidate": inspections.get("candidate", {}).get("review", {}),
    }
    packet.artifacts = [
        {"label": label, **artifact}
        for label, inspection in inspections.items()
        for artifact in list(inspection.get("artifacts", []) or [])
    ]
    packet.artifact_summary = _summarize_artifacts(packet.artifacts)
    deltas = _metric_deltas(metric_table.get("base", {}), metric_table.get("candidate", {}))
    metric_status = _comparison_metric_status(metrics, metric_table, deltas, inspections)
    packet.comparison = {
        "metric_names": list(metrics),
        "query_names": list(comparison_query_names),
        "metrics": metric_table,
        "deltas": deltas,
        "metric_status": metric_status,
        "inspection_statuses": inspection_statuses,
        "summary": _summarize_comparison(metric_status, inspection_statuses=inspection_statuses),
    }
    return _write_packet(packet, outdir)


def compare_outputs(
    base_output_dir: str | Path,
    candidate_output_dir: str | Path,
    *,
    metric_names: tuple[str, ...] | list[str] | None = None,
    max_rows: int = 50,
    max_value_bytes: int | None = None,
    max_result_bytes: int | None = None,
    protected: bool | None = None,
) -> dict[str, Any]:
    """Compare semantic metrics from two completed runs without executing or writing."""

    metrics = tuple(metric_names or DEFAULT_COMPARISON_METRICS)
    query_names = _comparison_query_names(metrics)
    inspections = {
        "base": inspect_output(
            base_output_dir,
            query_names=query_names,
            max_rows=max_rows,
            semantic_metric_names=metrics,
            write_packet=False,
            max_value_bytes=max_value_bytes,
            max_result_bytes=max_result_bytes,
            protected=protected,
        ),
        "candidate": inspect_output(
            candidate_output_dir,
            query_names=query_names,
            max_rows=max_rows,
            semantic_metric_names=metrics,
            write_packet=False,
            max_value_bytes=max_value_bytes,
            max_result_bytes=max_result_bytes,
            protected=protected,
        ),
    }
    metric_table = {label: _extract_metric_values(data, metrics) for label, data in inspections.items()}
    deltas = _metric_deltas(metric_table["base"], metric_table["candidate"])
    statuses = {label: str(data.get("status") or "") for label, data in inspections.items()}
    metric_status = _comparison_metric_status(metrics, metric_table, deltas, inspections)
    summary = _summarize_comparison(metric_status, inspection_statuses=statuses)
    return {
        "status": "completed" if summary["complete"] else "partial",
        "metric_names": list(metrics),
        "query_names": list(query_names),
        "metrics": metric_table,
        "deltas": deltas,
        "metric_status": metric_status,
        "inspection_statuses": statuses,
        "summary": summary,
    }


def prepare_recipe_config(
    recipe: AgentTaskRecipe,
    *,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    resource_profile: str = "config",
    output_writer: OutputWriter | None = None,
) -> dict[str, Any]:
    root = _repo_root()
    config_path = (root / recipe.config_path).resolve()
    if output_dir is None and output_root is None and output_writer is not None:
        output_dir = Path(output_writer.logical_root)
    if output_dir is None and output_root is not None:
        output_dir = _lexical_path(output_root) / recipe.recipe_id
    if output_dir is None:
        data = _load_yaml(config_path)
        output_dir = data.get("outputs", {}).get("output_dir", "")
    return _prepare_config(
        config_path, output_dir, label=recipe.recipe_id,
        resource_profile=resource_profile, output_writer=output_writer
    )


def _prepare_config(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    label: str,
    resource_profile: str = "config",
    output_writer: OutputWriter | None = None,
) -> dict[str, Any]:
    source = Path(config_path).expanduser().resolve()
    outdir = _lexical_path(output_dir)
    binding = _bind_output_writer(output_writer, outdir)
    data = _load_yaml(source)
    if resource_profile != "config":
        data = apply_resource_profile_to_config_dict(data, resource_profile)
        data.setdefault("outputs", {}).setdefault("stats", {})["print_summary"] = False
    data.setdefault("outputs", {})["output_dir"] = str(outdir)
    data.setdefault("outputs", {}).setdefault("review", {})
    data["outputs"]["review"]["enabled"] = True
    data["outputs"]["review"].setdefault("detail", "standard")
    task_config = outdir / "agent_task_config.yaml"
    serialized = yaml.safe_dump(data, sort_keys=False, allow_unicode=False).encode("utf-8")
    physical_task_config = binding.physical(task_config) if binding is not None else task_config
    if binding is None:
        outdir.mkdir(parents=True, exist_ok=True)
        task_config.write_bytes(serialized)
    else:
        binding.ensure_directory(outdir)
        binding.write_bytes(task_config, serialized)
    task_bytes = serialized
    task_digest = hashlib.sha256(task_bytes).hexdigest()
    normalization_policy = ConfigPathPolicy.default(
        config_path=source,
        workspace_root=_repo_root(),
        # A protected MCP task materializes its generated config beneath the
        # private descriptor-backed staging root.  The loader must be allowed
        # to read that exact generated file while retaining the source config
        # root; no caller-supplied roots are added here.
        read_roots=(source.parent, physical_task_config.parent),
        write_roots=(outdir,),
        allow_config_dir_writes=False,
    )
    prepared_cfg = _AGENT_TASK_EXECUTION_SERVICE.load_config(
        physical_task_config,
        path_policy=normalization_policy,
    )
    normalized = scenario_config_from_dict(data, source_path=source, path_policy=normalization_policy).to_dict()
    normalized_bytes = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    raw_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
    _PREPARED_CONFIG_CACHE[str(physical_task_config)] = (task_digest, prepared_cfg)
    _EXPECTED_PREPARED_CONFIG_DIGESTS[str(physical_task_config)] = task_digest
    return {
        "label": str(label),
        "source_config": str(source),
        "config_path": str(task_config),
        "output_dir": str(outdir),
        "review_enabled": True,
        "resource_profile": resource_profile,
        "source_config_sha256": raw_sha256,
        "normalized_config_sha256": normalized_sha256,
        "validation_id": f"oel-m4-validation-v1:{normalized_sha256}",
    }


def _guarded_prepared_config(config_path: str | Path) -> Any | None:
    """Return a prepared config, failing closed if its exact bytes changed."""

    path = _lexical_path(config_path)
    path_text = str(path)
    expected_digest = _EXPECTED_PREPARED_CONFIG_DIGESTS.get(path_text)
    if expected_digest is None:
        return None
    cached = _PREPARED_CONFIG_CACHE.get(path_text)
    if cached is None:
        raise RuntimeError(f"prepared agent-task config cache is missing for {path}")
    try:
        current_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"prepared agent-task config is unavailable: {path}") from exc
    if current_digest != expected_digest or cached[0] != expected_digest:
        raise RuntimeError(f"prepared agent-task config changed after validation: {path}")
    return cached[1]


def _discard_prepared_config(config_path: str | Path) -> None:
    path_text = str(_lexical_path(config_path))
    _PREPARED_CONFIG_CACHE.pop(path_text, None)
    _EXPECTED_PREPARED_CONFIG_DIGESTS.pop(path_text, None)


def _run_prepared_config(
    config_path: str | Path,
    *,
    step_callback: Any | None = None,
) -> dict[str, Any]:
    """Execute an exactly guarded parsed task config, falling back to the file API."""

    path = _lexical_path(config_path)
    cfg = _guarded_prepared_config(path)
    if run_simulation_config_file is not _RUN_SIMULATION_CONFIG_FILE_ORIGINAL:
        try:
            if step_callback is None:
                return run_simulation_config_file(path)
            return run_simulation_config_file(path, step_callback=step_callback)
        finally:
            if cfg is not None:
                _discard_prepared_config(path)
    if cfg is None:
        return run_simulation_config_file(path, step_callback=step_callback)
    try:
        study_type = _AGENT_TASK_EXECUTION_SERVICE.study_type(cfg)
        if study_type == "single_run":
            payload = _AGENT_TASK_EXECUTION_SERVICE.run_single(cfg, step_callback=step_callback)
            return _AGENT_TASK_EXECUTION_SERVICE.wrap_single_file_payload(
                payload=payload,
                cfg=cfg,
                config_path=path,
            )
        return _AGENT_TASK_EXECUTION_SERVICE.run_session_payload(
            cfg,
            source_path=path,
            step_callback=step_callback,
        )
    finally:
        _discard_prepared_config(path)


def _run_saved_queries(
    workspace: ReviewWorkspace,
    query_names: tuple[str, ...],
    *,
    max_rows: int,
    max_value_bytes: int | None = None,
    max_result_bytes: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in query_names:
        saved = get_saved_review_query(name)
        if saved is None:
            rows.append(
                {
                    "name": name,
                    "known": False,
                    "reason": "unknown_saved_query",
                    "status": "unknown_query",
                }
            )
            continue
        try:
            query_kwargs: dict[str, Any] = {
                "max_rows": max_rows,
                "max_vm_steps": saved.max_vm_steps,
            }
            if max_value_bytes is not None:
                query_kwargs["max_value_bytes"] = max_value_bytes
            if max_result_bytes is not None:
                query_kwargs["max_result_bytes"] = max_result_bytes
            result = workspace.query(saved.sql, **query_kwargs)
            empty_result = int(result.row_count) == 0
            rows.append(
                {
                    "name": saved.name,
                    "known": True,
                    "description": saved.description,
                    "sql": saved.sql,
                    "maturity": saved.maturity,
                    "source_tables": list(saved.source_tables),
                    "allow_empty": saved.allow_empty,
                    "max_vm_steps": saved.max_vm_steps,
                    "empty_result": empty_result,
                    "empty_result_allowed": saved.allow_empty,
                    "empty_result_unexpected": empty_result and not saved.allow_empty,
                    "status": "ok",
                    "columns": result.columns,
                    "rows": result.rows,
                    "row_count": result.row_count,
                    "truncated": result.truncated,
                }
            )
        except ReviewQueryError as exc:
            rows.append(
                {
                    "name": saved.name,
                    "known": True,
                    "description": saved.description,
                    "sql": saved.sql,
                    "maturity": saved.maturity,
                    "source_tables": list(saved.source_tables),
                    "max_vm_steps": saved.max_vm_steps,
                    "allow_empty": saved.allow_empty,
                    "status": "failed",
                    "error": str(exc),
                    "failure_hints": [hint.to_dict() for hint in diagnose_failure(str(exc))],
                }
            )
    return rows


def _comparison_query_names(metric_names: tuple[str, ...]) -> tuple[str, ...]:
    names = ["run_metadata", "artifacts"]
    for metric_name in metric_names:
        metric = get_semantic_metric(metric_name)
        saved_query = str(getattr(metric, "saved_query", "") or "") if metric is not None else ""
        if saved_query and saved_query not in names:
            names.append(saved_query)
    return tuple(names)


def _summarize_query_rows(query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [str(row.get("name")) for row in query_rows if row.get("status") == "failed"]
    unknown = [str(row.get("name")) for row in query_rows if row.get("status") == "unknown_query"]
    unexpected_empty = [str(row.get("name")) for row in query_rows if bool(row.get("empty_result_unexpected"))]
    truncated = [str(row.get("name")) for row in query_rows if bool(row.get("truncated"))]
    return {
        "total": len(query_rows),
        "ok": sum(1 for row in query_rows if row.get("status") == "ok"),
        "failed": len(failed),
        "unknown": len(unknown),
        "unexpected_empty": len(unexpected_empty),
        "truncated": len(truncated),
        "failed_queries": failed,
        "unknown_queries": unknown,
        "unexpected_empty_queries": unexpected_empty,
        "truncated_queries": truncated,
        "evidence_complete": not failed and not unknown and not unexpected_empty and not truncated,
    }


def _make_plots_for_recipe(
    output_dir: str | Path,
    plot_recipe_ids: tuple[str, ...],
    *,
    style_name: str,
    logical_output_dir: str | Path | None = None,
    protected: bool | None = None,
) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    workspace_dir = _lexical_path(output_dir)
    target_root = _lexical_path(logical_output_dir if logical_output_dir is not None else output_dir)
    for recipe_id in plot_recipe_ids:
        try:
            recipe = get_plot_recipe(recipe_id)
            # A descriptor-bound renderer needs an explicit logical target so
            # the plotting API can stage the bytes and retain the logical path
            # in its returned artifact record.  The old implicit target path
            # is still used for unknown recipes so their normal diagnostic is
            # preserved.
            target = None
            if recipe is not None:
                target = target_root / "review" / "figures" / f"{recipe.artifact_id}.png"
            plots.append(
                create_plot(
                    workspace_dir,
                    recipe_id,
                    style_name=style_name,
                    path=target,
                    logical_output_dir=target_root if logical_output_dir is not None else None,
                    protected=protected,
                )
            )
        except Exception as exc:
            plots.append(
                {
                    "recipe_id": recipe_id,
                    "status": "failed",
                    "error": str(exc),
                    "failure_hints": [hint.to_dict() for hint in diagnose_failure(str(exc), output_dir=output_dir)],
                }
            )
    return plots


def _extract_metric_values(inspection: dict[str, Any], metric_names: tuple[str, ...]) -> dict[str, Any]:
    wanted = set(metric_names)
    values: dict[str, Any] = {}
    review = dict(inspection.get("review", {}) or {})
    for query in list(review.get("queries", []) or []):
        query_name = str(query.get("name", "") or "")
        rows = [dict(row or {}) for row in list(query.get("rows", []) or [])]
        for row in rows:
            metric_name = str(row.get("metric_name", "") or "")
            if metric_name in wanted and "value" in row:
                values[metric_name] = row.get("value")
            if query.get("name") == "rendezvous_closest_approach" and "closest_approach_km" in wanted:
                values.setdefault("closest_approach_km", row.get("range_km"))
            if query.get("name") == "rendezvous_closest_approach" and "closest_approach_time_s" in wanted:
                values.setdefault("closest_approach_time_s", row.get("time_s"))
        if query_name == "relative_final_state" and len(rows) == 1 and "final_range_rate_km_s" in wanted:
            values["final_range_rate_km_s"] = rows[0].get("range_rate_km_s")
        if query_name == "object_final_state" and len(rows) == 1:
            if "final_radius_km" in wanted:
                values["final_radius_km"] = rows[0].get("radius_km")
            if "final_speed_km_s" in wanted:
                values["final_speed_km_s"] = rows[0].get("speed_km_s")
        if query_name == "object_orbital_elements_first_last" and rows:
            object_ids = {str(row.get("object_id", "")) for row in rows}
            if len(object_ids) == 1 and "final_semi_major_axis_km" in wanted:
                final_row = max(rows, key=lambda row: float(row.get("time_s", 0.0) or 0.0))
                values["final_semi_major_axis_km"] = final_row.get("a_km")
        if query_name == "burn_command_summary" and "total_delta_v_m_s" in wanted:
            realized = [row.get("realized_delta_v_m_s") for row in rows]
            if realized and all(value is not None for value in realized):
                values["total_delta_v_m_s"] = sum(float(value) for value in realized)
    return values


def _metric_deltas(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for name in sorted(set(base) & set(candidate)):
        try:
            deltas[name] = float(candidate[name]) - float(base[name])
        except (TypeError, ValueError):
            continue
    return deltas


def _comparison_metric_status(
    metric_names: tuple[str, ...],
    metric_table: dict[str, dict[str, Any]],
    deltas: dict[str, Any],
    inspections: dict[str, Any],
) -> list[dict[str, Any]]:
    base = metric_table.get("base", {})
    candidate = metric_table.get("candidate", {})
    status_rows: list[dict[str, Any]] = []
    for name in metric_names:
        metric = get_semantic_metric(name)
        base_available = name in base
        candidate_available = name in candidate
        row: dict[str, Any] = {
            "name": name,
            "base_available": base_available,
            "candidate_available": candidate_available,
            "delta_available": name in deltas,
            "semantic_metric_known": metric is not None,
        }
        if metric is not None:
            query_status_by_label = _saved_query_statuses(inspections, metric.saved_query)
            row.update(
                {
                    "maturity": metric.maturity,
                    "source_tables": list(metric.source_tables),
                    "saved_query": metric.saved_query,
                    "query_status_by_label": query_status_by_label,
                }
            )
            if not base_available or not candidate_available:
                if (
                    metric.saved_query
                    and query_status_by_label
                    and all(status == "ok" for status in query_status_by_label.values())
                ):
                    row["reason"] = "no_scalar_reducer"
                else:
                    row["reason"] = "metric_unavailable"
            elif name not in deltas:
                row["reason"] = "non_numeric_delta"
        else:
            row["reason"] = "unknown_semantic_metric"
        status_rows.append(row)
    return status_rows


def _saved_query_statuses(inspections: dict[str, Any], query_name: str) -> dict[str, str]:
    if not query_name:
        return {}
    out: dict[str, str] = {}
    for label, inspection in inspections.items():
        queries = {
            str(query.get("name") or ""): dict(query)
            for query in list(dict(inspection.get("review", {}) or {}).get("queries", []) or [])
            if isinstance(query, dict)
        }
        query = queries.get(str(query_name))
        out[str(label)] = str(query.get("status") or "missing") if query is not None else "missing"
    return out


def _summarize_comparison(
    metric_status: list[dict[str, Any]],
    *,
    inspection_statuses: dict[str, str],
) -> dict[str, Any]:
    unknown = [str(row.get("name")) for row in metric_status if not bool(row.get("semantic_metric_known"))]
    missing_values = [
        str(row.get("name"))
        for row in metric_status
        if bool(row.get("semantic_metric_known"))
        and (not bool(row.get("base_available")) or not bool(row.get("candidate_available")))
    ]
    missing_deltas = [
        str(row.get("name"))
        for row in metric_status
        if bool(row.get("semantic_metric_known"))
        and bool(row.get("base_available"))
        and bool(row.get("candidate_available"))
        and not bool(row.get("delta_available"))
    ]
    partial_inspections = [
        str(label) for label, status in sorted(inspection_statuses.items()) if status and status != "completed"
    ]
    return {
        "total": len(metric_status),
        "unknown_metrics": unknown,
        "missing_value_metrics": missing_values,
        "missing_delta_metrics": missing_deltas,
        "partial_inspections": partial_inspections,
        "complete": not unknown and not missing_values and not missing_deltas and not partial_inspections,
    }


def _artifact_rows(query_rows: list[dict[str, Any]], *, output_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for query in query_rows:
        if query.get("name") not in {"artifacts", "workflow_artifacts"} or query.get("status") != "ok":
            continue
        for row in list(query.get("rows", []) or []):
            artifacts.append(_artifact_with_path_status(dict(row), output_dir=output_dir))
    return artifacts


def _workflow_artifacts(manifest: dict[str, Any], *, output_dir: Path) -> list[dict[str, Any]]:
    return [
        _artifact_with_path_status(dict(item), output_dir=output_dir)
        for item in list(manifest.get("artifacts", []) or [])
        if isinstance(item, dict)
    ]


def _artifacts_with_path_status(artifacts: list[dict[str, Any]], *, output_dir: Path) -> list[dict[str, Any]]:
    return [_artifact_with_path_status(dict(item), output_dir=output_dir) for item in artifacts]


def _partial_output_artifacts(output_dir: str | Path, *, maximum: int = 256) -> list[dict[str, Any]]:
    root = _lexical_path(output_dir)
    if not root.is_dir():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if len(artifacts) >= maximum:
            break
        if path.is_file() and not path.is_symlink():
            artifacts.append(
                _artifact_with_path_status(
                    {"artifact_id": f"partial_{len(artifacts):04d}", "path": str(path), "partial": True},
                    output_dir=root,
                )
            )
    return artifacts


def _artifact_with_path_status(artifact: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    raw_path = str(artifact.get("path") or "")
    if not raw_path:
        artifact["path_exists"] = False
        artifact["resolved_path"] = ""
        return artifact
    path = Path(raw_path).expanduser()
    resolved = path if path.is_absolute() else output_dir / path
    artifact["resolved_path"] = str(resolved)
    artifact["path_exists"] = resolved.exists()
    return artifact


def _summarize_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [
        str(item.get("artifact_id") or item.get("artifact_key") or item.get("path") or "unknown")
        for item in artifacts
        if item.get("path_exists") is False
    ]
    unknown = [
        str(item.get("artifact_id") or item.get("artifact_key") or item.get("path") or "unknown")
        for item in artifacts
        if "path_exists" not in item
    ]
    existing = sum(1 for item in artifacts if item.get("path_exists") is True)
    return {
        "total": len(artifacts),
        "existing": existing,
        "missing": len(missing),
        "path_status_unknown": len(unknown),
        "missing_artifacts": missing,
        "path_status_unknown_artifacts": unknown,
        "artifacts_complete": not missing and not unknown,
    }


def _maybe_workflow_manifest(output_dir: Path) -> dict[str, Any] | None:
    try:
        return load_workflow_manifest(output_dir)
    except Exception:
        return None


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload.get("run", {})
    return {
        "summary": dict(summary or {}),
        "analysis": dict(payload.get("analysis", {}) or {}),
        "monte_carlo": dict(payload.get("monte_carlo", {}) or {}),
    }


def _plot_artifact_dict(artifact: ReviewPlotArtifact, *, recipe: AgentPlotRecipe) -> dict[str, Any]:
    path = Path(artifact.path).expanduser()
    return {
        "recipe_id": recipe.recipe_id,
        "recipe_maturity": recipe.maturity,
        "source_tables": list(recipe.supported_tables),
        "semantic_metric_names": list(recipe.semantic_metric_names),
        "artifact_id": artifact.artifact_id,
        "path": str(path),
        "resolved_path": str(path),
        "path_exists": path.exists(),
        "relative_path": artifact.relative_path,
        "row_count": artifact.row_count,
        "truncated": artifact.truncated,
        "qa": dict(artifact.qa),
        "spec": asdict(artifact.spec),
        "status": "ok",
    }


def _summarize_plots(plots: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [
        str(item.get("recipe_id") or item.get("artifact_id") or "unknown")
        for item in plots
        if item.get("status") == "failed"
    ]
    missing = [
        str(item.get("artifact_id") or item.get("recipe_id") or item.get("path") or "unknown")
        for item in plots
        if item.get("path_exists") is False
    ]
    truncated = [
        str(item.get("artifact_id") or item.get("recipe_id") or "unknown")
        for item in plots
        if bool(item.get("truncated"))
    ]
    ok = sum(1 for item in plots if item.get("status") == "ok")
    return {
        "total": len(plots),
        "ok": ok,
        "failed": len(failed),
        "missing": len(missing),
        "truncated": len(truncated),
        "failed_plots": failed,
        "missing_plots": missing,
        "truncated_plots": truncated,
        "plots_complete": not failed and not missing and not truncated,
    }


def _summarize_packet_evidence(packet: EvidencePacket) -> dict[str, Any]:
    validation_ok = _validation_ok(packet.validation)
    review_complete = _review_evidence_complete(packet.review)
    artifacts_complete = _optional_complete(packet.artifact_summary, "artifacts_complete")
    plots_complete = _optional_complete(packet.plot_summary, "plots_complete")
    comparison_complete = _comparison_complete(packet.comparison)
    failure_hint_count = len(packet.failure_hints)
    ready = (
        packet.status == "completed"
        and bool(packet.run)
        and validation_ok is not False
        and review_complete is not False
        and artifacts_complete is not False
        and plots_complete is not False
        and comparison_complete is not False
        and failure_hint_count == 0
    )
    readiness_blockers: list[str] = []
    if packet.status != "completed":
        readiness_blockers.append(f"status:{packet.status}")
    if not packet.run:
        readiness_blockers.append("run_evidence_absent")
    if validation_ok is False:
        readiness_blockers.append("validation_failed")
    elif validation_ok is None:
        readiness_blockers.append("validation_evidence_absent")
    if review_complete is False:
        readiness_blockers.append("review_evidence_incomplete")
    if artifacts_complete is False:
        readiness_blockers.append("artifacts_incomplete")
    if plots_complete is False:
        readiness_blockers.append("plots_incomplete")
    if comparison_complete is False:
        readiness_blockers.append("comparison_incomplete")
    if failure_hint_count:
        readiness_blockers.append("failure_hints_present")
    return {
        "status": packet.status,
        "packet_mode": "execution" if bool(packet.run) else "inspection_only",
        "validation_ok": validation_ok,
        "review_evidence_complete": review_complete,
        "artifacts_complete": artifacts_complete,
        "plots_complete": plots_complete,
        "comparison_complete": comparison_complete,
        "failure_hint_count": failure_hint_count,
        "caveat_count": len(packet.caveats),
        "ready_to_cite": ready,
        "readiness_blockers": readiness_blockers,
    }


def _validation_ok(validation: dict[str, Any]) -> bool | None:
    if not validation:
        return None
    if "ok" in validation:
        return bool(validation.get("ok"))
    reports = [item for item in validation.values() if isinstance(item, dict) and "ok" in item]
    if not reports:
        return None
    return all(bool(item.get("ok")) for item in reports)


def _review_evidence_complete(review: dict[str, Any]) -> bool | None:
    summaries = _collect_query_summaries(review)
    if not summaries:
        return None
    return all(bool(item.get("evidence_complete")) for item in summaries)


def _collect_query_summaries(value: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if isinstance(value, dict):
        summary = value.get("query_summary")
        if isinstance(summary, dict):
            summaries.append(summary)
        for item in value.values():
            summaries.extend(_collect_query_summaries(item))
    elif isinstance(value, list):
        for item in value:
            summaries.extend(_collect_query_summaries(item))
    return summaries


def _optional_complete(summary: dict[str, Any], key: str) -> bool | None:
    if not summary:
        return None
    if key not in summary:
        return None
    return bool(summary.get(key))


def _comparison_complete(comparison: dict[str, Any]) -> bool | None:
    if not comparison:
        return None
    summary = comparison.get("summary")
    if not isinstance(summary, dict) or "complete" not in summary:
        return None
    return bool(summary.get("complete"))


def _write_packet(
    packet: EvidencePacket,
    output_dir: str | Path,
    *,
    output_writer: OutputWriter | None = None,
) -> dict[str, Any]:
    outdir = _lexical_path(output_dir)
    binding = _bind_output_writer(output_writer, outdir)
    if binding is None:
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        binding.ensure_directory(outdir)
    packet.evidence_summary = _summarize_packet_evidence(packet)
    payload = _json_safe(packet.to_dict())
    packet_path = outdir / "agent_evidence_packet.json"
    payload["packet_path"] = str(packet_path)
    serialized = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    if binding is None:
        packet_path.write_bytes(serialized)
    else:
        binding.write_bytes(packet_path, serialized)
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scenario config must be a YAML mapping: {path}")
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _workspace() -> SimulationWorkspace:
    root = _repo_root()
    return SimulationWorkspace(workspace_root=root, read_roots=(root,), write_roots=(root,))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
