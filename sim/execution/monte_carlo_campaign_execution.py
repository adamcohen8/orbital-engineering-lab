# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *
from .monte_carlo_checkpoints import *
from sim.execution.thread_campaign import ThreadCampaignExecutor, ensure_thread_campaign_compatible
from .monte_carlo_preparation import *
from .validation import raise_for_monte_carlo_preflight_errors, validate_prepared_monte_carlo_runs
from .campaign_retention import compact_iteration_result, policy_from_config
from .workers import initialize_campaign_worker_transport


def _validated_reused_monte_carlo_preparation(
    *,
    cfg: SimulationScenarioConfig,
    outdir: Path,
    prepared: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Accept only the private, complete preparation shape for this campaign.

    The caller retains the exact dictionaries produced earlier in the same
    execution.  Cheap structural and output-ownership checks prevent a public
    caller from accidentally reusing a partial or cross-campaign collection;
    per-run ``config_hash`` values remain the checkpoint content guard.
    """

    expected_count = int(cfg.monte_carlo.iterations)
    if len(prepared) != expected_count:
        raise ValueError("reused Monte Carlo preparation has the wrong run count")
    expected_indices = list(range(expected_count))
    actual_indices = [int(item.get("iteration", -1)) for item in prepared]
    if actual_indices != expected_indices:
        raise ValueError("reused Monte Carlo preparation has inconsistent iteration ordering")
    for item in prepared:
        iteration = int(item["iteration"])
        config_dict = item.get("config_dict")
        if not isinstance(config_dict, dict) or not str(item.get("config_hash", "") or ""):
            raise ValueError("reused Monte Carlo preparation is incomplete")
        output_dir = str(dict(config_dict.get("outputs", {}) or {}).get("output_dir", "") or "")
        if Path(output_dir).expanduser().resolve() != (outdir / f"mc_run_{iteration:04d}").resolve():
            raise ValueError("reused Monte Carlo preparation belongs to a different output root")
    return prepared


def run_serial_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
    collect_payload: bool = False,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    relative_range_writer: MonteCarloRelativeRangePlotWriter | None = None,
    resource_governor: ResourceGovernor | None = None,
    checkpoint: bool = True,
    initial_completed: dict[int, dict[str, Any]] | None = None,
    _prepared: list[dict[str, Any]] | None = None,
    _validated_configs: list[SimulationScenarioConfig | None] | None = None,
) -> dict[str, Any]:
    require_pro_feature(FEATURE_CAMPAIGNS)
    prepared = (
        prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
        if _prepared is None
        else _validated_reused_monte_carlo_preparation(
            cfg=cfg,
            outdir=outdir,
            prepared=_prepared,
        )
    )
    if _validated_configs is None:
        preflight, validated_configs = validate_prepared_monte_carlo_runs(
            prepared=prepared,
            strict_plugins=strict_plugins,
            source_path=source_path if path_policy is not None else None,
            path_policy=path_policy,
        )
        raise_for_monte_carlo_preflight_errors(preflight)
    else:
        if len(_validated_configs) != len(prepared) or any(run_cfg is None for run_cfg in _validated_configs):
            raise ValueError("reused Monte Carlo preflight has the wrong run count or invalid configs")
        validated_configs = _validated_configs
    from sim.execution.metrics import normalize_study_metric_specs

    metric_paths = [str(spec["path"]) for spec in normalize_study_metric_specs(cfg.analysis.metrics)]
    metric_paths += [str(entry.get("metric", entry.get("path", ""))) for entry in
                     _metric_gate_entries(dict(cfg.outputs.monte_carlo or {}).get("gates", {}))]
    policy = policy_from_config(cfg.to_dict())
    collect_payload = collect_payload or _metric_paths_need_payload(metric_paths)
    completed: dict[int, dict[str, Any]] = dict(initial_completed or {})
    total_iters = int(cfg.monte_carlo.iterations)
    completed_count = len(completed)

    for prepared_index, item in enumerate(prepared):
        iteration = int(item["iteration"])
        if iteration in completed:
            continue
        if checkpoint:
            cached = _load_mc_checkpoint(outdir, iteration, str(item.get("config_hash", "")))
            if cached is not None:
                completed[iteration] = cached
                if relative_range_writer is not None:
                    marker = getattr(relative_range_writer, "mark_checkpoint_omitted", None)
                    if callable(marker):
                        marker(iteration)
                completed_count += 1
                if batch_callback is not None:
                    try:
                        batch_callback(completed_count, total_iters)
                    except Exception as exc:
                        _raise_if_cancellation(exc)
                        logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                        batch_callback = None
                continue
        if resource_governor is not None:
            resource_governor.wait_for_capacity(
                context=f"Monte Carlo iteration {iteration + 1}/{total_iters}",
                include_load=False,
            )
        ci = validated_configs[prepared_index]
        if ci is None:  # pragma: no cover - guarded by preflight
            raise RuntimeError(f"Monte Carlo iteration {iteration} has no validated config")
        run_payload = _compat_run_single_config(ci, step_callback=step_callback)
        if relative_range_writer is not None and relative_range_writer.should_collect_iteration(iteration):
            relative_range_writer.add_run(
                iteration=iteration,
                series=_compat_relative_range_series(run_payload),
            )
        completed[iteration] = {
            "iteration": iteration,
            "summary": run_payload["summary"],
            "closest_approach_km": _compat_closest_approach(run_payload),
            "derived": {
                "relative_motion": relative_motion_summary_from_run_payload(run_payload),
            },
        }
        rel = dict(completed[iteration]["derived"]["relative_motion"])
        for key in ("initial_range_km", "final_range_km", "final_relative_speed_m_s", "max_relative_speed_m_s"):
            if key in rel:
                completed[iteration]["derived"][key] = rel.get(key)
        if collect_payload:
            completed[iteration]["payload"] = run_payload
        if policy:
            compact_iteration_result(
                run_payload, completed[iteration], policy=policy, metric_paths=metric_paths,
                output_dir=str(dict(item["config_dict"].get("outputs", {}) or {}).get("output_dir", "")),
            )
        if checkpoint:
            _write_mc_checkpoint(outdir, iteration, completed[iteration], str(item.get("config_hash", "")))
        completed_count += 1
        if batch_callback is not None:
            try:
                batch_callback(completed_count, total_iters)
            except Exception as exc:
                _raise_if_cancellation(exc)
                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                batch_callback = None

    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": False,
        "parallel_fallback_reason": None,
        "checkpoint": {
            "enabled": bool(checkpoint),
            "checkpoint_dir": str(_mc_checkpoint_dir(outdir)),
            "resumed_count": int(sum(1 for item in completed.values() if bool(item.get("resumed_from_checkpoint")))),
            "written_count": int(sum(1 for item in completed.values() if not bool(item.get("resumed_from_checkpoint")))),
        },
    }


def run_monte_carlo_runs(
    *,
    cfg: SimulationScenarioConfig,
    root: dict[str, Any],
    outdir: Path,
    strict_plugins: bool,
    mc_out_cfg: dict[str, Any],
    source_path: str | Path | None = None,
    path_policy: Any | None = None,
    step_callback: StepCallback | None = None,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    relative_range_writer: MonteCarloRelativeRangePlotWriter | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    total_iters = int(cfg.monte_carlo.iterations)
    requested_backend = backend
    if requested_backend is None:
        requested_backend = getattr(cfg.monte_carlo, "parallel_backend", "process_pool")
    parallel_backend = str(requested_backend or "process_pool").strip().lower()
    if parallel_backend not in {"process_pool", "thread_pool"}:
        raise ValueError("Monte Carlo parallel backend must be 'process_pool' or 'thread_pool'.")
    parallel_enabled = bool(cfg.monte_carlo.parallel_enabled)
    max_workers_cfg = int(cfg.monte_carlo.parallel_workers or 0)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    parallel_workers = max_workers_cfg if max_workers_cfg > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_iters, 1)))
    parallel_active = bool(parallel_enabled and total_iters > 1)
    parallel_fallback_reason: str | None = None
    prepared = prepare_monte_carlo_runs(cfg=cfg, root=root, outdir=outdir)
    preflight, validated_configs = validate_prepared_monte_carlo_runs(
        prepared=prepared,
        strict_plugins=strict_plugins,
        source_path=source_path if path_policy is not None else None,
        path_policy=path_policy,
    )
    raise_for_monte_carlo_preflight_errors(preflight)
    hash_by_idx = {int(item["iteration"]): str(item.get("config_hash", "")) for item in prepared}
    completed: dict[int, dict[str, Any]] = {}
    from sim.execution.metrics import normalize_study_metric_specs

    metric_specs = normalize_study_metric_specs(cfg.analysis.metrics)
    metric_paths = [str(spec["path"]) for spec in metric_specs]
    metric_paths += [str(entry.get("metric", entry.get("path", ""))) for entry in
                     _metric_gate_entries(mc_out_cfg.get("gates", {}))]
    policy = policy_from_config(root)
    collect_payload = _metric_gates_need_payload(mc_out_cfg.get("gates", {}) or {}) or _metric_paths_need_payload(
        [str(spec["path"]) for spec in metric_specs]
    )
    checkpoint = checkpoint_enabled(cfg, mc_out_cfg)
    checkpoint_status_before = (
        monte_carlo_checkpoint_status(cfg=cfg, root=root, _prepared=prepared)
        if checkpoint
        else {}
    )
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
                label=f"Monte Carlo iteration {int(item['iteration'])}",
                allow_outer_analysis=True,
            )
        resource_governor.wait_for_capacity(context="threaded parallel Monte Carlo launch")
        for item in prepared:
            iteration = int(item["iteration"])
            if checkpoint:
                cached = _load_mc_checkpoint(outdir, iteration, str(item.get("config_hash", "")))
                if cached is not None:
                    completed[iteration] = cached
                    if relative_range_writer is not None:
                        marker = getattr(relative_range_writer, "mark_checkpoint_omitted", None)
                        if callable(marker):
                            marker(iteration)
        tasks = [
            {
                "iteration": item["iteration"],
                "config_dict": item["config_dict"],
                "strict_plugins": strict_plugins,
                "source_path": source_path if path_policy is not None else None,
                "path_policy": path_policy,
                "progress_emit_every": (
                    1
                    if step_callback is not None
                    else int(mc_out_cfg.get("parallel_progress_emit_every_steps", 20) or 20)
                ),
                "collect_relative_range_series": bool(
                    relative_range_writer is not None
                    and relative_range_writer.should_collect_iteration(int(item["iteration"]))
                ),
                "collect_payload": collect_payload,
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
                        result = dict(future.result())
                        if relative_range_writer is not None:
                            relative_range_writer.add_run(
                                iteration=idx,
                                series=result.pop("relative_range_series", None),
                            )
                        completed[idx] = result
                        if checkpoint:
                            _write_mc_checkpoint(outdir, idx, result, hash_by_idx.get(idx, ""))
                        if batch_callback is not None:
                            try:
                                batch_callback(len(completed), total_iters)
                            except Exception as exc:
                                _raise_if_cancellation(exc)
                                logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                                batch_callback = None
                    step_callback, batch_progress_callback = _drain_parallel_progress_events(
                        executor.progress_queue,
                        step_callback=step_callback,
                        batch_progress_callback=batch_progress_callback,
                        callback_label="threaded Monte Carlo",
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
                        callback_label="threaded Monte Carlo",
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
            resource_governor.wait_for_capacity(context="parallel Monte Carlo launch")
            try:
                cancel_event = mp.Event()
                if step_callback is not None or batch_progress_callback is not None:
                    manager = mp.Manager()
                    progress_queue = manager.Queue()
            except (OSError, PermissionError, EOFError) as exc:
                logger.warning(
                    "Parallel Monte Carlo cancellation transport is unavailable; falling back to serial execution: %s",
                    exc,
                )
                manager = None
                progress_queue = None
                cancel_event = None
                raise
            for item in prepared:
                iteration = int(item["iteration"])
                if checkpoint:
                    cached = _load_mc_checkpoint(outdir, iteration, str(item.get("config_hash", "")))
                    if cached is not None:
                        completed[iteration] = cached
                        if relative_range_writer is not None:
                            marker = getattr(relative_range_writer, "mark_checkpoint_omitted", None)
                            if callable(marker):
                                marker(iteration)
            tasks = [
                {
                    "iteration": item["iteration"],
                    "config_dict": item["config_dict"],
                    "strict_plugins": strict_plugins,
                    "source_path": source_path if path_policy is not None else None,
                    "path_policy": path_policy,
                    "progress_emit_every": (
                        1
                        if step_callback is not None
                        else int(mc_out_cfg.get("parallel_progress_emit_every_steps", 20) or 20)
                    ),
                    "collect_relative_range_series": bool(
                        relative_range_writer is not None
                        and relative_range_writer.should_collect_iteration(int(item["iteration"]))
                    ),
                    "collect_payload": collect_payload,
                    "metric_paths": metric_paths if policy else [],
                }
                for item in prepared
                if int(item["iteration"]) not in completed
            ]
            if not tasks:
                parallel_active = False
            pool_kwargs = {
                "initializer": initialize_campaign_worker_transport,
                "initargs": (progress_queue, cancel_event),
            }
            if tasks:
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
                                result = dict(fut.result())
                            except (
                                OSError,
                                PermissionError,
                                NotImplementedError,
                                EOFError,
                                BrokenProcessPool,
                                ResourcePressureError,
                            ):
                                raise
                            else:
                                if relative_range_writer is not None:
                                    relative_range_writer.add_run(
                                        iteration=idx,
                                        series=result.pop("relative_range_series", None),
                                    )
                                completed[idx] = result
                                if checkpoint:
                                    _write_mc_checkpoint(outdir, idx, result, hash_by_idx.get(idx, ""))
                                if batch_callback is not None:
                                    try:
                                        batch_callback(len(completed), total_iters)
                                    except Exception as exc:
                                        _raise_if_cancellation(exc)
                                        logger.warning("Disabling Monte Carlo callback after runtime error: %s", exc)
                                        batch_callback = None
                        step_callback, batch_progress_callback = _drain_parallel_progress_events(
                            progress_queue,
                            step_callback=step_callback,
                            batch_progress_callback=batch_progress_callback,
                            callback_label="Monte Carlo",
                        )
                except BaseException:
                    if cancel_event is not None:
                        cancel_event.set()
                    abort_process_pool(executor)
                    raise
                else:
                    executor.shutdown(wait=True, cancel_futures=False)
        except (
            OSError,
            PermissionError,
            NotImplementedError,
            EOFError,
            BrokenProcessPool,
            ResourcePressureError,
        ) as exc:
            parallel_active = False
            parallel_fallback_reason = format_parallel_fallback_reason(exc)
            logger.warning("Parallel Monte Carlo unavailable, falling back to serial execution: %s", exc)
        finally:
            step_callback, batch_progress_callback = _drain_parallel_progress_events(
                progress_queue,
                step_callback=step_callback,
                batch_progress_callback=batch_progress_callback,
                callback_label="Monte Carlo",
            )
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            restore_env_vars(thread_env_prev)
            restore_hierarchical_worker_env(hierarchy_env_prev)

    if not parallel_active:
        serial_result = run_serial_monte_carlo_runs(
            cfg=cfg,
            root=root,
            outdir=outdir,
            strict_plugins=strict_plugins,
            source_path=source_path if path_policy is not None else None,
            path_policy=path_policy,
            collect_payload=collect_payload,
            step_callback=step_callback,
            batch_callback=batch_callback,
            relative_range_writer=relative_range_writer,
            resource_governor=resource_governor,
            checkpoint=checkpoint,
            initial_completed=completed,
            _prepared=prepared,
            _validated_configs=validated_configs,
        )
        prepared = list(serial_result.get("prepared", []) or [])
        completed = dict(serial_result.get("completed", {}) or {})

    checkpoint_meta = {
        "enabled": bool(checkpoint),
        "checkpoint_dir": str(_mc_checkpoint_dir(outdir)),
        "resumed_count": int(sum(1 for item in completed.values() if bool(dict(item or {}).get("resumed_from_checkpoint")))),
        "completed_count": int(len(completed)),
        "matching_before_count": int(dict(checkpoint_status_before).get("matching_count", 0) or 0),
        "stale_before_count": int(dict(checkpoint_status_before).get("stale_count", 0) or 0),
        "legacy_before_count": int(dict(checkpoint_status_before).get("legacy_count", 0) or 0),
    }
    return {
        "prepared": prepared,
        "completed": completed,
        "parallel_active": bool(parallel_active),
        "parallel_backend": parallel_backend if parallel_active else "serial",
        "parallel_workers": int(parallel_workers if parallel_active else 1),
        "hierarchical_execution": hierarchy.payload(),
        "parallel_fallback_reason": parallel_fallback_reason,
        "checkpoint": checkpoint_meta,
        "resource_wait": resource_governor.telemetry(),
    }

__all__ = [name for name in globals() if not name.startswith("__")]
