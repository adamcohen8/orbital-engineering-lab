from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from sim.config import SimulationScenarioConfig, scenario_config_from_dict, validate_scenario_plugins
from sim.execution.campaign_common import _drain_parallel_progress_events, _raise_if_cancellation
from sim.execution.campaigns import _config_fingerprint, _json_safe
from sim.execution.hierarchical import (
    apply_hierarchical_worker_env,
    plan_hierarchical_execution,
    restore_hierarchical_worker_env,
)
from sim.execution.parameter_paths import set_parameter_path_value, validate_parameter_path_claims
from sim.execution.thread_campaign import (
    ThreadCampaignBackendUnavailable,
    ThreadCampaignExecutor,
    ensure_thread_campaign_compatible,
)
from sim.pro_features import FEATURE_CAMPAIGNS, require_pro_feature
from sim.resource_limits import ResourceGovernor, ResourcePressureError, checkpoint_enabled
from sim.single_run import _run_single_config
from sim.utils.io import write_json
from sim.utils.parallel import (
    abort_process_pool,
    format_parallel_fallback_reason,
    initialize_worker_progress_queue,
    iter_bounded_futures,
)

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)

_PARALLEL_STARTUP_ERRORS = (
    OSError,
    PermissionError,
    NotImplementedError,
    EOFError,
    BrokenProcessPool,
    ResourcePressureError,
)
_PARALLEL_FUTURE_TRANSPORT_ERRORS = (EOFError, BrokenProcessPool)


def _sensitivity_checkpoint_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "sensitivity_checkpoints"


def _sensitivity_checkpoint_path(output_dir: str | Path, iteration: int) -> Path:
    return _sensitivity_checkpoint_dir(output_dir) / f"iteration_{int(iteration):04d}.json"


def _load_sensitivity_checkpoint(output_dir: str | Path, iteration: int, config_hash: str) -> dict[str, Any] | None:
    path = _sensitivity_checkpoint_path(output_dir, iteration)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable sensitivity checkpoint %s: %s", path, exc)
        return None
    if not isinstance(raw, dict) or str(raw.get("config_hash", "") or "") != str(config_hash):
        return None
    result = raw.get("result")
    if not isinstance(result, dict):
        return None
    stored_iteration = result.get("iteration")
    if (
        type(stored_iteration) is not int
        or stored_iteration != int(iteration)
        or result.get("status") != "completed"
    ):
        return None
    resumed_result = dict(result)
    resumed_result["resumed_from_checkpoint"] = True
    return resumed_result


def _write_sensitivity_checkpoint(output_dir: str | Path, iteration: int, result: dict[str, Any], config_hash: str) -> None:
    path = _sensitivity_checkpoint_path(output_dir, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        str(path),
        {
            "checkpoint_version": 1,
            "iteration": int(iteration),
            "config_hash": str(config_hash),
            "result": _json_safe(result),
        },
    )


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _prepare_analysis_run_config(
    cdict: dict[str, Any],
    *,
    outdir: Path,
    run_prefix: str,
    iteration: int,
) -> dict[str, Any]:
    from sim.execution.campaign_retention import configure_iteration

    mode = str(cdict.get("outputs", {}).get("mode", "interactive"))
    if mode == "interactive":
        cdict.setdefault("outputs", {})["mode"] = "save"
    cdict.setdefault("outputs", {})["output_dir"] = str(outdir / f"{run_prefix}_{iteration:04d}")
    configure_iteration(cdict)
    return cdict


def _prepare_oaat_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    iteration = 0
    for param in cfg.analysis.sensitivity.parameters:
        parameter_path = str(param.parameter_path)
        if not parameter_path:
            raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be non-empty.")
        values = list(param.values or [])
        if not values:
            raise ValueError(f"analysis.sensitivity.parameters[{parameter_path!r}] must provide at least one value.")
        for value_index, value in enumerate(values):
            cdict = _prepare_analysis_run_config(
                deepcopy(root),
                outdir=outdir,
                run_prefix="sensitivity_run",
                iteration=iteration,
            )
            set_parameter_path_value(cdict, parameter_path, value)
            prepared.append(
                {
                    "iteration": iteration,
                    "parameter_path": parameter_path,
                    "parameter_value": value,
                    "value_index": value_index,
                    "sampled_parameters": {parameter_path: value},
                    "config_dict": cdict,
                }
            )
            iteration += 1
    return prepared


def _lhs_parameter_value(param: Any, unit_sample: float) -> float:
    unit = float(np.clip(float(unit_sample), 1.0e-12, 1.0 - 1.0e-12))
    distribution = str(getattr(param, "distribution", "uniform") or "uniform").strip().lower()
    if distribution == "uniform":
        low = getattr(param, "low", None)
        high = getattr(param, "high", None)
        if low is None or high is None:
            raise ValueError(f"LHS parameter '{param.parameter_path}' requires low/high for uniform distribution.")
        return float(low) + unit * (float(high) - float(low))
    if distribution == "normal":
        mean = getattr(param, "mean", None)
        std = getattr(param, "std", None)
        if mean is None or std is None or float(std) <= 0.0:
            raise ValueError(f"LHS parameter '{param.parameter_path}' requires mean/std with std > 0.")
        return float(NormalDist(mu=float(mean), sigma=float(std)).inv_cdf(unit))
    raise ValueError(f"Unsupported LHS distribution '{distribution}' for parameter '{param.parameter_path}'.")


def _prepare_lhs_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    params = list(cfg.analysis.sensitivity.parameters or [])
    samples = int(cfg.analysis.sensitivity.samples or 0)
    if samples <= 0:
        raise ValueError("analysis.sensitivity.samples must be > 0 for method='lhs'.")
    if not params:
        raise ValueError("analysis.sensitivity.parameters must contain at least one parameter.")

    rng = np.random.default_rng(int(cfg.analysis.sensitivity.seed))
    sampled_columns: dict[str, np.ndarray] = {}
    for param in params:
        unit_samples = (rng.permutation(samples).astype(float) + rng.random(samples)) / float(samples)
        sampled_columns[str(param.parameter_path)] = np.array(
            [_lhs_parameter_value(param, unit) for unit in unit_samples],
            dtype=float,
        )

    prepared: list[dict[str, Any]] = []
    for iteration in range(samples):
        cdict = _prepare_analysis_run_config(
            deepcopy(root),
            outdir=outdir,
            run_prefix="sensitivity_lhs_run",
            iteration=iteration,
        )
        sampled_parameters: dict[str, Any] = {}
        for param in params:
            parameter_path = str(param.parameter_path)
            value = float(sampled_columns[parameter_path][iteration])
            set_parameter_path_value(cdict, parameter_path, value)
            sampled_parameters[parameter_path] = value
        prepared.append(
            {
                "iteration": iteration,
                "parameter_path": None,
                "parameter_value": None,
                "value_index": iteration,
                "sampled_parameters": sampled_parameters,
                "config_dict": cdict,
            }
        )
    return prepared


def _prepare_two_parameter_grid_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
) -> list[dict[str, Any]]:
    params = list(cfg.analysis.sensitivity.parameters or [])
    if len(params) != 2:
        raise ValueError(
            "analysis.sensitivity.parameters must contain exactly two parameters for method='two_parameter_grid'."
        )
    for param in params:
        if not str(param.parameter_path):
            raise ValueError("analysis.sensitivity.parameters[*].parameter_path must be non-empty.")
        if not list(param.values or []):
            raise ValueError(
                f"analysis.sensitivity.parameters[{param.parameter_path!r}] must provide at least one value."
            )

    p0, p1 = params
    values0 = list(p0.values or [])
    values1 = list(p1.values or [])
    prepared: list[dict[str, Any]] = []
    iteration = 0
    for i, value0 in enumerate(values0):
        for j, value1 in enumerate(values1):
            cdict = _prepare_analysis_run_config(
                deepcopy(root),
                outdir=outdir,
                run_prefix="sensitivity_grid_run",
                iteration=iteration,
            )
            path0 = str(p0.parameter_path)
            path1 = str(p1.parameter_path)
            set_parameter_path_value(cdict, path0, value0)
            set_parameter_path_value(cdict, path1, value1)
            prepared.append(
                {
                    "iteration": iteration,
                    "parameter_path": f"{path0}|{path1}",
                    "parameter_value": {"x": value0, "y": value1},
                    "value_index": iteration,
                    "grid_indices": [i, j],
                    "sampled_parameters": {path0: value0, path1: value1},
                    "config_dict": cdict,
                }
            )
            iteration += 1
    return prepared


def prepare_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Any,
    sensitivity_method: str,
) -> list[dict[str, Any]]:
    method = str(sensitivity_method or "one_at_a_time").strip().lower()
    out_path = Path(outdir)
    validate_parameter_path_claims(
        root,
        [
            (f"sensitivity[{index}]", str(parameter.parameter_path), False)
            for index, parameter in enumerate(cfg.analysis.sensitivity.parameters or [])
        ],
    )
    if method == "lhs":
        return _prepare_lhs_sensitivity_runs(cfg=cfg, root=root, outdir=out_path)
    if method == "two_parameter_grid":
        return _prepare_two_parameter_grid_runs(cfg=cfg, root=root, outdir=out_path)
    return _prepare_oaat_sensitivity_runs(cfg=cfg, root=root, outdir=out_path)


def validate_prepared_sensitivity_runs(
    *,
    prepared: list[dict[str, Any]],
    strict_plugins: bool,
    import_plugins: bool = True,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
) -> dict[str, Any]:
    preflight, _validated = _validate_and_parse_prepared_sensitivity_runs(
        prepared=prepared,
        strict_plugins=strict_plugins,
        import_plugins=import_plugins,
        source_path=source_path,
        path_policy=path_policy,
    )
    return preflight


def _validate_and_parse_prepared_sensitivity_runs(
    *,
    prepared: list[dict[str, Any]],
    strict_plugins: bool,
    import_plugins: bool = True,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
) -> tuple[dict[str, Any], list[tuple[str, SimulationScenarioConfig | None]]]:
    """Validate once and retain exact normalized configs for serial execution."""

    errors: list[dict[str, Any]] = []
    validated: list[tuple[str, SimulationScenarioConfig | None]] = []
    for item in prepared:
        iteration = int(item.get("iteration", 0))
        config_dict = dict(item.get("config_dict", {}) or {})
        config_hash = _config_fingerprint(config_dict)
        try:
            run_cfg = scenario_config_from_dict(
                config_dict,
                source_path=source_path,
                path_policy=path_policy,
            )
            plugin_errors = (
                validate_scenario_plugins(run_cfg, import_plugins=import_plugins) if strict_plugins else []
            )
        except Exception as exc:
            validated.append((config_hash, None))
            errors.append(
                {
                    "iteration": iteration,
                    "parameter_path": item.get("parameter_path"),
                    "parameter_value": item.get("parameter_value"),
                    "sampled_parameters": dict(item.get("sampled_parameters", {}) or {}),
                    "error": str(exc),
                }
            )
            continue
        validated.append((config_hash, run_cfg))
        for plugin_error in plugin_errors:
            errors.append(
                {
                    "iteration": iteration,
                    "parameter_path": item.get("parameter_path"),
                    "parameter_value": item.get("parameter_value"),
                    "sampled_parameters": dict(item.get("sampled_parameters", {}) or {}),
                    "error": str(plugin_error),
                }
            )
    return (
        {
            "run_count": int(len(prepared)),
            "valid_run_count": int(len(prepared) - len({int(e["iteration"]) for e in errors})),
            "error_count": int(len(errors)),
            "errors": errors,
        },
        validated,
    )


def _format_preflight_error(error: dict[str, Any]) -> str:
    sampled = dict(error.get("sampled_parameters", {}) or {})
    sampled_text = ", ".join(f"{k}={v!r}" for k, v in sampled.items())
    if not sampled_text:
        sampled_text = f"{error.get('parameter_path')}={error.get('parameter_value')!r}"
    return f"run {error.get('iteration')}: {sampled_text}: {error.get('error')}"


def raise_for_sensitivity_preflight_errors(preflight: dict[str, Any]) -> None:
    errors = list(preflight.get("errors", []) or [])
    if not errors:
        return
    lines = [_format_preflight_error(dict(error)) for error in errors[:8]]
    if len(errors) > len(lines):
        lines.append(f"... {len(errors) - len(lines)} more error(s)")
    raise ValueError("Generated sensitivity run validation failed:\n- " + "\n- ".join(lines))


def _failed_sensitivity_run(iteration: int, exc: BaseException) -> dict[str, Any]:
    return {
        "iteration": int(iteration),
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "summary": {},
        "closest_approach_km": float("nan"),
        "payload": {"summary": {}},
    }


def run_sensitivity_runs(
    *,
    cfg: SimulationScenarioConfig,
    prepared: list[dict[str, Any]],
    strict_plugins: bool,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    _preflight: dict[str, Any] | None = None,
    _validated_configs: list[tuple[str, SimulationScenarioConfig | None]] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    from sim.execution.campaign_retention import compact_iteration_result, policy_from_config
    from sim.execution.metrics import closest_approach_from_run_payload
    from sim.execution.workers import restore_env_vars, run_mc_iteration_from_dict, set_parallel_worker_thread_limits
    from sim.reporting.sensitivity import analysis_metric_specs

    parallel_enabled = bool(cfg.analysis.execution.parallel_enabled)
    requested_backend = backend
    if requested_backend is None:
        requested_backend = getattr(cfg.analysis.execution, "parallel_backend", "process_pool")
    parallel_backend = str(requested_backend or "process_pool").strip().lower()
    if parallel_backend not in {"process_pool", "thread_pool"}:
        raise ValueError("Sensitivity parallel backend must be 'process_pool' or 'thread_pool'.")
    failure_policy = str(getattr(cfg.analysis.execution, "failure_policy", "fail_fast") or "fail_fast").strip().lower()
    continue_on_error = failure_policy == "continue"
    max_workers_cfg = int(cfg.analysis.execution.parallel_workers or 0)
    total_iters = int(len(prepared))
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_iters, 1)))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    completed: dict[int, dict[str, Any]] = {}
    if _preflight is None or _validated_configs is None:
        preflight, validated_configs = _validate_and_parse_prepared_sensitivity_runs(
            prepared=prepared,
            strict_plugins=strict_plugins,
            source_path=source_path if path_policy is not None else None,
            path_policy=path_policy,
        )
    else:
        preflight = dict(_preflight)
        validated_configs = _validated_configs
        if len(validated_configs) != len(prepared):
            raise ValueError("reused sensitivity preparation has the wrong run count")
    raise_for_sensitivity_preflight_errors(preflight)
    output_dir = Path(cfg.outputs.output_dir)
    policy = policy_from_config(cfg.to_dict())
    metric_paths = [str(spec["path"]) for spec in analysis_metric_specs(cfg)]
    config_hash_by_idx = {
        int(item.get("iteration", 0)): validated_configs[index][0]
        for index, item in enumerate(prepared)
    }
    checkpoint = checkpoint_enabled(cfg)
    if checkpoint:
        for item in prepared:
            iteration = int(item.get("iteration", 0))
            cached = _load_sensitivity_checkpoint(output_dir, iteration, config_hash_by_idx.get(iteration, ""))
            if cached is not None:
                completed[iteration] = cached
    resource_governor = ResourceGovernor(cfg, emit=lambda msg: logger.warning("%s", msg))
    hierarchy = plan_hierarchical_execution(
        task_roots=[dict(item.get("config_dict", {}) or {}) for item in prepared],
        task_count=total_iters,
        requested_campaign_workers=parallel_workers,
        profile=resource_governor.profile,
    )
    parallel_workers = int(hierarchy.campaign_workers)
    if resource_governor.profile.force_serial or parallel_workers <= 1:
        parallel_active = False
        if parallel_enabled:
            parallel_fallback_reason = f"hierarchical_planner:{hierarchy.reason}"

    if parallel_active and parallel_backend == "thread_pool":
        for item in prepared:
            ensure_thread_campaign_compatible(
                dict(item.get("config_dict", {}) or {}),
                label=f"Sensitivity iteration {int(item['iteration'])}",
                allow_outer_analysis=True,
            )
        resource_governor.wait_for_capacity(context="threaded parallel sensitivity launch")
        tasks = [
            {
                "iteration": item["iteration"],
                "config_dict": item["config_dict"],
                "strict_plugins": strict_plugins,
                "source_path": source_path if path_policy is not None else None,
                "path_policy": path_policy,
                "progress_emit_every": 1 if step_callback is not None else 20,
                "collect_payload": True,
                "metric_paths": metric_paths if policy else [],
                "worker_budget": dict(hierarchy.payload()),
            }
            for item in prepared
            if int(item["iteration"]) not in completed
        ]
        if not tasks:
            parallel_active = False
        else:
            executor = ThreadCampaignExecutor(max_workers=parallel_workers)
            try:
                for future, task in executor.iter_completed(run_mc_iteration_from_dict, tasks):
                    if future is not None:
                        idx = int(task["iteration"])
                        try:
                            completed[idx] = future.result()
                            completed[idx]["status"] = "completed"
                        except ThreadCampaignBackendUnavailable:
                            raise
                        except Exception as exc:
                            if not continue_on_error:
                                raise
                            completed[idx] = _failed_sensitivity_run(idx, exc)
                        if checkpoint and completed[idx].get("status") == "completed":
                            _write_sensitivity_checkpoint(
                                output_dir, idx, completed[idx], config_hash_by_idx.get(idx, "")
                            )
                        if batch_callback is not None:
                            try:
                                batch_callback(len(completed), total_iters)
                            except Exception as exc:
                                _raise_if_cancellation(exc)
                                logger.warning("Disabling analysis callback after runtime error: %s", exc)
                                batch_callback = None
                    step_callback, batch_progress_callback = _drain_parallel_progress_events(
                        executor.progress_queue,
                        step_callback=step_callback,
                        batch_progress_callback=batch_progress_callback,
                        callback_label="threaded analysis",
                    )
            except BaseException:
                executor.cancel()
                raise
            finally:
                try:
                    step_callback, batch_progress_callback = _drain_parallel_progress_events(
                        executor.progress_queue,
                        step_callback=step_callback,
                        batch_progress_callback=batch_progress_callback,
                        callback_label="threaded analysis",
                    )
                finally:
                    executor.shutdown(wait_for_running=True)

    if parallel_active and parallel_backend == "process_pool":
        manager = None
        progress_queue = None
        cancel_event = None
        thread_env_prev = set_parallel_worker_thread_limits(default_threads="1")
        hierarchy_env_prev = apply_hierarchical_worker_env(hierarchy)
        try:
            resource_governor.wait_for_capacity(context="parallel sensitivity launch")
            try:
                manager = mp.Manager()
                cancel_event = manager.Event()
                if step_callback is not None or batch_progress_callback is not None:
                    progress_queue = manager.Queue()
            except (OSError, PermissionError, EOFError) as exc:
                logger.warning(
                    "Parallel sensitivity cancellation transport is unavailable; falling back to serial execution: %s",
                    exc,
                )
                manager = None
                progress_queue = None
                cancel_event = None
                raise
            tasks = [
                {
                    "iteration": item["iteration"],
                    "config_dict": item["config_dict"],
                    "strict_plugins": strict_plugins,
                    "source_path": source_path if path_policy is not None else None,
                    "path_policy": path_policy,
                    "progress_emit_every": 1 if step_callback is not None else 20,
                    "cancel_event": cancel_event,
                    "collect_payload": True,
                    "metric_paths": metric_paths if policy else [],
                }
                for item in prepared
                if int(item["iteration"]) not in completed
            ]
            if not tasks:
                parallel_active = False
            else:
                pool_kwargs = {}
                if progress_queue is not None:
                    pool_kwargs = {
                        "initializer": initialize_worker_progress_queue,
                        "initargs": (progress_queue,),
                    }
                executor = ProcessPoolExecutor(max_workers=parallel_workers, **pool_kwargs)
                try:
                    for fut, task in iter_bounded_futures(
                        executor,
                        run_mc_iteration_from_dict,
                        tasks,
                        max_in_flight=2 * parallel_workers,
                    ):
                        if fut is not None:
                            idx = int(task["iteration"])
                            try:
                                completed[idx] = fut.result()
                                completed[idx]["status"] = "completed"
                            except _PARALLEL_FUTURE_TRANSPORT_ERRORS:
                                raise
                            except Exception as exc:
                                if not continue_on_error:
                                    raise
                                completed[idx] = _failed_sensitivity_run(idx, exc)
                            if checkpoint and completed[idx].get("status") == "completed":
                                _write_sensitivity_checkpoint(
                                    output_dir, idx, completed[idx], config_hash_by_idx.get(idx, "")
                                )
                            if batch_callback is not None:
                                try:
                                    batch_callback(len(completed), total_iters)
                                except Exception as exc:
                                    _raise_if_cancellation(exc)
                                    logger.warning("Disabling analysis callback after runtime error: %s", exc)
                                    batch_callback = None
                        step_callback, batch_progress_callback = _drain_parallel_progress_events(
                            progress_queue,
                            step_callback=step_callback,
                            batch_progress_callback=batch_progress_callback,
                            callback_label="analysis",
                        )
                except BaseException:
                    if cancel_event is not None:
                        cancel_event.set()
                    abort_process_pool(executor)
                    raise
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
        except _PARALLEL_STARTUP_ERRORS as exc:
            parallel_active = False
            parallel_fallback_reason = format_parallel_fallback_reason(exc)
            logger.warning("Parallel sensitivity analysis unavailable, falling back to serial execution: %s", exc)
        finally:
            step_callback, batch_progress_callback = _drain_parallel_progress_events(
                progress_queue,
                step_callback=step_callback,
                batch_progress_callback=batch_progress_callback,
                callback_label="analysis",
            )
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            restore_env_vars(thread_env_prev)
            restore_hierarchical_worker_env(hierarchy_env_prev)

    if not parallel_active:
        completed_count = 0
        for prepared_index, prepared_run in enumerate(prepared):
            iteration = int(prepared_run["iteration"])
            if iteration in completed:
                completed_count += 1
                if batch_callback is not None:
                    try:
                        batch_callback(completed_count, total_iters)
                    except Exception as exc:
                        _raise_if_cancellation(exc)
                        logger.warning("Disabling analysis callback after runtime error: %s", exc)
                        batch_callback = None
                continue
            resource_governor.wait_for_capacity(
                context=f"sensitivity run {iteration + 1}/{total_iters}",
                include_load=False,
            )
            run_cfg = validated_configs[prepared_index][1]
            if run_cfg is None:  # pragma: no cover - guarded by preflight
                raise RuntimeError(f"sensitivity run {iteration} has no validated config")
            try:
                run_payload = _run_single_config(run_cfg, step_callback=step_callback)
                completed[iteration] = {
                    "iteration": iteration,
                    "status": "completed",
                    "summary": dict(run_payload.get("summary", {}) or {}),
                    "closest_approach_km": closest_approach_from_run_payload(run_payload),
                    "payload": run_payload,
                }
                if policy:
                    compact_iteration_result(
                        run_payload, completed[iteration], policy=policy, metric_paths=metric_paths,
                        output_dir=str(dict(prepared_run["config_dict"].get("outputs", {}) or {}).get("output_dir", "")),
                    )
                if checkpoint:
                    _write_sensitivity_checkpoint(
                        output_dir, iteration, completed[iteration], config_hash_by_idx.get(iteration, "")
                    )
            except Exception as exc:
                if not continue_on_error:
                    raise
                completed[iteration] = _failed_sensitivity_run(iteration, exc)
            completed_count += 1
            if batch_callback is not None:
                try:
                    batch_callback(completed_count, total_iters)
                except Exception as exc:
                    _raise_if_cancellation(exc)
                    logger.warning("Disabling analysis callback after runtime error: %s", exc)
                    batch_callback = None

    return {
        "completed": completed,
        "parallel_active": bool(parallel_active),
        "parallel_backend": parallel_backend if parallel_active else "serial",
        "parallel_workers": int(parallel_workers if parallel_active else 1),
        "hierarchical_execution": hierarchy.payload(),
        "parallel_fallback_reason": parallel_fallback_reason,
        "preflight": preflight,
        "failure_policy": failure_policy,
        "resource_wait": resource_governor.telemetry(),
        "checkpoint": {
            "enabled": bool(checkpoint),
            "checkpoint_dir": str(_sensitivity_checkpoint_dir(output_dir)),
            "resumed_count": int(sum(1 for item in completed.values() if bool(item.get("resumed_from_checkpoint")))),
            "completed_count": int(len(completed)),
        },
    }


def run_sensitivity_analysis(
    *,
    config_path: str | Path,
    cfg: SimulationScenarioConfig,
    path_policy: Any | None = None,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    require_pro_feature(FEATURE_CAMPAIGNS)
    sensitivity_method = str(cfg.analysis.sensitivity.method).strip().lower()
    if sensitivity_method not in {"one_at_a_time", "lhs", "two_parameter_grid"}:
        raise ValueError("analysis.sensitivity.method must be one of: one_at_a_time, lhs, two_parameter_grid.")
    if not cfg.analysis.sensitivity.parameters:
        raise ValueError("analysis.sensitivity.parameters must contain at least one parameter.")

    root = cfg.to_dict()
    outdir = Path(cfg.outputs.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    from sim.reporting.sensitivity import analysis_metric_specs, analysis_metrics, extract_analysis_metrics

    metric_specs = analysis_metric_specs(cfg)
    metric_paths = analysis_metrics(cfg)
    prepared = prepare_sensitivity_runs(
        cfg=cfg,
        root=root,
        outdir=outdir,
        sensitivity_method=sensitivity_method,
    )
    from sim.execution.campaign_retention import policy_from_config, preflight_campaign_retention

    preflight_campaign_retention(policy_from_config(root), len(prepared))
    strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
    preflight, validated_configs = _validate_and_parse_prepared_sensitivity_runs(
        prepared=prepared,
        strict_plugins=strict_plugins,
        source_path=config_path if path_policy is not None else None,
        path_policy=path_policy,
    )
    raise_for_sensitivity_preflight_errors(preflight)

    baseline: dict[str, Any] | None = None
    baseline_mode = str(getattr(cfg.analysis.baseline, "mode", "none") or "none").strip().lower()
    baseline_summary_json = str(cfg.analysis.baseline.summary_json or "").strip()
    if baseline_mode == "file":
        bpath = Path(baseline_summary_json)
        if not bpath.is_absolute():
            bpath = Path(config_path).resolve().parent / bpath
        if not baseline_summary_json:
            raise ValueError("analysis.baseline.mode=file requires analysis.baseline.summary_json.")
        if not bpath.is_file():
            raise FileNotFoundError(f"Sensitivity baseline summary not found: {bpath}")
        baseline_payload = _load_json_file(bpath)
        if baseline_payload is None:
            raise ValueError(f"Sensitivity baseline summary is not a valid JSON object: {bpath}")
        baseline_summary = dict(baseline_payload.get("summary", baseline_payload.get("run", {})) or {})
        if not baseline_summary:
            raise ValueError(f"Sensitivity baseline contains no summary/run object: {bpath}")
        baseline = {
            "source": "file",
            "path": str(bpath),
            "summary": baseline_summary,
            "metrics": extract_analysis_metrics(baseline_payload, metric_specs),
        }
    elif baseline_mode == "run" or (baseline_mode == "none" and bool(cfg.analysis.baseline.enabled)):
        baseline_root = deepcopy(root)
        mode = str(baseline_root.get("outputs", {}).get("mode", "interactive"))
        if mode == "interactive":
            baseline_root.setdefault("outputs", {})["mode"] = "save"
        baseline_root.setdefault("outputs", {})["output_dir"] = str(outdir / "sensitivity_baseline")
        baseline_root.setdefault("analysis", {})["enabled"] = False
        if policy_from_config(root) and policy_from_config(root).get("mode", "stats") != "full":
            from sim.execution.campaign_retention import configure_iteration

            configure_iteration(baseline_root)
            baseline_root["outputs"]["review"] = {"enabled": False}
        baseline_payload = _run_single_config(scenario_config_from_dict(baseline_root))
        baseline = {
            "source": "run",
            "output_dir": str(outdir / "sensitivity_baseline"),
            "summary": dict(baseline_payload.get("summary", {}) or {}),
            "metrics": extract_analysis_metrics(baseline_payload, metric_specs),
        }

    sensitivity_result = run_sensitivity_runs(
        cfg=cfg,
        prepared=prepared,
        strict_plugins=strict_plugins,
        source_path=config_path if path_policy is not None else None,
        path_policy=path_policy,
        step_callback=step_callback,
        batch_callback=batch_callback,
        batch_progress_callback=batch_progress_callback,
        _preflight=preflight,
        _validated_configs=validated_configs,
        backend=backend,
    )
    completed = dict(sensitivity_result.get("completed", {}) or {})
    parallel_active = bool(sensitivity_result.get("parallel_active", False))
    parallel_workers = int(sensitivity_result.get("parallel_workers", 1) or 1)
    parallel_backend = str(sensitivity_result.get("parallel_backend", "serial") or "serial")
    parallel_fallback_reason = sensitivity_result.get("parallel_fallback_reason")
    preflight = dict(sensitivity_result.get("preflight", {}) or {})
    failure_policy = str(
        sensitivity_result.get("failure_policy", getattr(cfg.analysis.execution, "failure_policy", "fail_fast"))
    )
    checkpoint_meta = dict(sensitivity_result.get("checkpoint", {}) or {})

    from sim.reporting.ai_reports import write_ai_report_artifacts
    from sim.reporting.sensitivity import build_sensitivity_report_payload, write_sensitivity_summary_artifact

    agg = build_sensitivity_report_payload(
        cfg=cfg,
        config_path=config_path,
        prepared=prepared,
        completed=completed,
        baseline=baseline,
        metric_paths=metric_paths,
        metric_specs=metric_specs,
        sensitivity_method=sensitivity_method,
        parallel_enabled=bool(cfg.analysis.execution.parallel_enabled),
        parallel_active=parallel_active,
        parallel_workers=parallel_workers,
        parallel_fallback_reason=parallel_fallback_reason,
        preflight=preflight,
        failure_policy=failure_policy,
    )
    agg["hierarchical_execution"] = dict(
        sensitivity_result.get("hierarchical_execution", {}) or {}
    )
    agg["parallel_backend"] = parallel_backend
    from sim.execution.campaign_retention import policy_from_config, retire_iteration_review, write_campaign_dataset

    retention_policy = policy_from_config(root)
    if retention_policy:
        records: list[dict[str, Any]] = []
        for item, run in zip(prepared, agg["runs"], strict=True):
            iteration = int(item["iteration"])
            result = dict(completed.get(iteration, {}) or {})
            run_dir = Path(dict(item["config_dict"].get("outputs", {}) or {})["output_dir"])
            keep_full = (retention_policy.get("mode", "stats") == "full"
                         or iteration in retention_policy.get("full_iterations", [])
                         or bool(retention_policy.get("full_on_failure", False) and run["status"] != "completed"))
            review_path = run_dir / "review" / "run.sqlite"
            if keep_full and run["status"] == "completed" and not review_path.is_file():
                raise FileNotFoundError(f"Retained full campaign review store missing: {review_path}")
            if not keep_full:
                retire_iteration_review(
                    run_dir, expected_sha256=dict(result.get("retained_evidence", {}) or {}).get("source_review_sha256")
                )
            records.append({
                "iteration": iteration, "seed": dict(item["config_dict"].get("metadata", {}) or {}).get("seed"),
                "sampled_parameters": dict(item.get("sampled_parameters", {}) or {}),
                "config_hash": _config_fingerprint(item["config_dict"]),
                "status": run["status"], "error": run.get("error"), "metrics": run.get("metrics", {}),
                "retained_evidence": dict(result.get("retained_evidence", {}) or {}),
                "full_review": str(review_path.relative_to(outdir))
                if keep_full and run["status"] == "completed" and review_path.is_file() else None,
            })
        agg.setdefault("artifacts", {}).update(
            write_campaign_dataset(outdir, policy=retention_policy, records=records)
        )
    if checkpoint_meta:
        agg["checkpoint"] = checkpoint_meta
    agg = write_ai_report_artifacts(
        cfg=cfg,
        config_path=config_path,
        outdir=outdir,
        payload=agg,
        payload_kind="sensitivity",
    )

    return write_sensitivity_summary_artifact(outdir=outdir, payload=agg)
