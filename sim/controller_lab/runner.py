# ruff: noqa: F401,F403,F405,I001
from .bench_common import *
from .config import *
from .variants import *
from .cases import *
from .criteria import *
from .diagnostics import *
from .objectives import *
from .optimization import *
from .engineering import render_closed_loop_response_plots, write_failure_reduction, write_suite_engineering_exports
from sim.utils.parallel import _drain_parallel_progress_events, _raise_if_cancellation
from sim.execution.thread_campaign import ThreadCampaignExecutor, ensure_thread_campaign_compatible

def run_controller_bench(
    config_path: str | Path,
    *,
    compare_names: list[str] | None = None,
    show_progress: bool = True,
    step_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    require_pro_feature(FEATURE_CONTROLLER_BENCH)
    resolved_config_path = str(Path(config_path).expanduser().resolve())
    suite = load_controller_bench_config(config_path)
    preflight = validate_controller_bench_config(config_path, compare_names=compare_names)
    if not bool(preflight.get("passed", False)):
        raise _controller_bench_preflight_error(list(preflight.get("errors", []) or []))
    if suite.optimization.enabled:
        return _run_controller_bench_optimization(
            suite,
            compare_names=compare_names,
            source_config_path=resolved_config_path,
            show_progress=show_progress,
        )
    selected = _select_controller_bench_variants(suite, compare_names)

    scheduled_runs = [(case, variant) for case in suite.cases for variant in selected]
    runs_by_index: dict[int, dict[str, Any]] = {}
    use_tqdm = bool(show_progress) and _should_use_tqdm()
    bench_bar = None
    plain_progress = None if use_tqdm or not show_progress else _make_plain_progress_reporter("Controller Bench")
    total_runs = len(scheduled_runs)
    parallel_requested = bool(suite.parallel_enabled and total_runs > 1)
    default_workers = max(1, (os.cpu_count() or 1) - 1)
    configured_workers = int(max(suite.parallel_workers, 0))
    parallel_workers = configured_workers if configured_workers > 0 else default_workers
    parallel_workers = max(1, min(parallel_workers, max(total_runs, 1)))
    parallel_active = False
    parallel_fallback_reason: str | None = None
    hierarchy: dict[str, Any] | None = None
    hierarchy_plan = None
    if parallel_requested:
        task_roots = [load_simulation_yaml(case.config_path).to_dict() for case, _variant in scheduled_runs]
        hierarchy_plan = plan_hierarchical_execution(
            task_roots=task_roots,
            task_count=total_runs,
            requested_campaign_workers=parallel_workers,
            profile=restrictive_profile_for_task_roots(task_roots),
        )
        hierarchy = hierarchy_plan.payload()
        parallel_workers = int(hierarchy_plan.campaign_workers)
        if parallel_workers <= 1:
            parallel_fallback_reason = f"hierarchical_planner:{hierarchy_plan.reason}"
    try:
        if use_tqdm and scheduled_runs:
            try:
                from tqdm.auto import tqdm  # type: ignore

                bench_bar = tqdm(
                    total=len(scheduled_runs),
                    desc="Controller Bench",
                    unit="run",
                    dynamic_ncols=True,
                )
            except Exception:
                bench_bar = None
                plain_progress = _make_plain_progress_reporter("Controller Bench")

        if parallel_requested and parallel_workers > 1 and suite.parallel_backend == "thread_pool":
            for case, _variant in scheduled_runs:
                ensure_thread_campaign_compatible(
                    load_simulation_yaml(case.config_path).to_dict(),
                    label=f"controller bench case {case.name!r}",
                )
            tasks = [
                {
                    "index": index,
                    "suite": suite,
                    "case": case,
                    "variant": variant,
                    "progress_emit_every": 20,
                    "worker_budget": dict(hierarchy_plan.payload()),
                }
                for index, (case, variant) in enumerate(scheduled_runs)
            ]
            executor = ThreadCampaignExecutor(max_workers=parallel_workers)
            try:
                parallel_active = True
                completed_count = 0
                for future, _task in executor.iter_completed(_run_single_bench_case_worker, tasks):
                    if future is not None:
                        item = future.result()
                        run = dict(item.get("run", {}) or {})
                        index = int(item.get("index", 0))
                        runs_by_index[index] = run
                        completed_count += 1
                        detail = f"{run.get('case_name', 'case')} :: {run.get('variant_name', 'variant')}"
                        if bench_bar is not None:
                            bench_bar.set_postfix_str(detail)
                            bench_bar.update(1)
                        elif plain_progress is not None:
                            plain_progress(completed_count, total_runs, detail)
                    step_callback, _unused_progress = _drain_parallel_progress_events(
                        executor.progress_queue,
                        step_callback=step_callback,
                        batch_progress_callback=None,
                        callback_label="controller bench",
                    )
            except BaseException:
                executor.cancel()
                raise
            finally:
                try:
                    step_callback, _unused_progress = _drain_parallel_progress_events(
                        executor.progress_queue,
                        step_callback=step_callback,
                        batch_progress_callback=None,
                        callback_label="controller bench",
                    )
                finally:
                    executor.shutdown(wait_for_running=True)

        if parallel_requested and parallel_workers > 1 and suite.parallel_backend == "process_pool":
            thread_env_prev = set_parallel_worker_thread_limits(default_threads="1")
            hierarchy_env_prev = apply_hierarchical_worker_env(hierarchy_plan)
            parallel_error: BaseException | None = None
            try:
                tasks = [
                    {
                        "index": index,
                        "suite": suite,
                        "case": case,
                        "variant": variant,
                    }
                    for index, (case, variant) in enumerate(scheduled_runs)
                ]
                with ProcessPoolExecutor(max_workers=parallel_workers) as ex:
                    parallel_active = True
                    completed_count = 0
                    for fut, _task in iter_bounded_futures(
                        ex,
                        _run_single_bench_case_worker,
                        tasks,
                        max_in_flight=2 * parallel_workers,
                    ):
                        if fut is not None:
                            try:
                                item = fut.result()
                            except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
                                parallel_error = exc
                            else:
                                run = dict(item.get("run", {}) or {})
                                index = int(item.get("index", 0))
                                runs_by_index[index] = run
                                completed_count += 1
                                detail = f"{run.get('case_name', 'case')} :: {run.get('variant_name', 'variant')}"
                                if bench_bar is not None:
                                    bench_bar.set_postfix_str(detail)
                                    bench_bar.update(1)
                                elif plain_progress is not None:
                                    plain_progress(completed_count, total_runs, detail)
                    if parallel_error is not None:
                        raise parallel_error
            except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
                parallel_active = False
                parallel_fallback_reason = format_parallel_fallback_reason(exc)
            finally:
                restore_env_vars(thread_env_prev)
                restore_hierarchical_worker_env(hierarchy_env_prev)

        if not parallel_active:
            completed_count = len(runs_by_index)
            for index, (case, variant) in enumerate(scheduled_runs):
                if index in runs_by_index:
                    continue
                detail = f"{case.name} :: {variant.name}"
                if bench_bar is not None:
                    bench_bar.set_postfix_str(detail)
                elif plain_progress is not None:
                    plain_progress(completed_count + 1, total_runs, detail)
                runs_by_index[index], _payload = _run_single_bench_case_with_payload(
                    suite, case, variant, step_callback=step_callback
                )
                completed_count += 1
                if bench_bar is not None:
                    bench_bar.update(1)
    finally:
        if bench_bar is not None:
            bench_bar.close()
    runs = [runs_by_index[idx] for idx in range(total_runs)]

    variant_summaries: list[dict[str, Any]] = []
    for variant in selected:
        subset = [run for run in runs if run["variant_name"] == variant.name]
        metric_names: list[str] = []
        objective_names: list[str] = []
        for run in subset:
            for metric_name in dict(run.get("metrics", {}) or {}).keys():
                if metric_name not in metric_names:
                    metric_names.append(metric_name)
            for objective_result in list(run.get("objective_results", []) or []):
                objective_name = str(objective_result.get("name", "") or "").strip()
                if objective_name and objective_name not in objective_names:
                    objective_names.append(objective_name)
        metric_means: dict[str, float] = {}
        for metric_name in metric_names:
            values = []
            for run in subset:
                value = dict(run.get("metrics", {}) or {}).get(metric_name)
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value_f):
                    continue
                values.append(value_f)
            if values:
                metric_means[metric_name] = float(sum(values) / len(values))
        objective_pass_rates: dict[str, float] = {}
        for objective_name in objective_names:
            objective_hits = 0
            objective_passes = 0
            for run in subset:
                for objective_result in list(run.get("objective_results", []) or []):
                    if str(objective_result.get("name", "") or "").strip() != objective_name:
                        continue
                    objective_hits += 1
                    if bool(objective_result.get("passed", False)):
                        objective_passes += 1
            if objective_hits > 0:
                objective_pass_rates[objective_name] = float(objective_passes) / float(objective_hits)
        run_count = len(subset)
        passed_runs = sum(1 for run in subset if bool(run.get("passed", False)))
        variant_summaries.append(
            {
                "variant_name": variant.name,
                "description": variant.description,
                "run_count": run_count,
                "passed_runs": passed_runs,
                "pass_rate": (float(passed_runs) / float(run_count)) if run_count > 0 else 0.0,
                "metric_means": metric_means,
                "objective_pass_rates": objective_pass_rates,
            }
        )

    result = {
        "suite_name": suite.suite_name,
        "description": suite.description,
        "config_path": resolved_config_path,
        "controller_target": {
            "object_id": suite.controller_target.object_id,
            "slot": suite.controller_target.slot,
        },
        "variants": [_variant_report_entry(variant) for variant in selected],
        "cases": [
            {
                "name": case.name,
                "config_path": str(case.config_path),
                "description": case.description,
                "objectives": [
                    {
                        "name": objective.name or objective.kind,
                        "kind": objective.kind,
                    }
                    for objective in (case.objectives if case.objectives else suite.objectives)
                ],
            }
            for case in suite.cases
        ],
        "envelope": deepcopy(dict(suite.envelope_manifest or {})),
        "runs": runs,
        "variant_summaries": variant_summaries,
        "execution": {
            "parallel_enabled": bool(parallel_active),
            "parallel_requested": bool(parallel_requested),
            "parallel_workers": int(parallel_workers if parallel_active else 1),
            "parallel_fallback_reason": parallel_fallback_reason,
            "hierarchical_execution": hierarchy,
        },
        "plot_mode": suite.plot_mode,
        "leaderboards": {
            "relative_rendezvous": _build_relative_rendezvous_leaderboards(runs, selected),
            "rocket_orbit_insertion": _build_rocket_insertion_leaderboards(runs, selected),
        },
        "constraint_summary": _aggregate_constraint_summary(runs),
        "semantic_lint": dict(preflight.get("semantic_lint", {}) or {}),
    }
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"] = {}
    render_closed_loop_response_plots(result)
    plot_artifacts = render_controller_bench_visualizations(result, suite.output_dir, suite.plot_mode)
    if plot_artifacts:
        result["artifacts"].update(plot_artifacts)
    if suite.engineering.enabled and suite.engineering.failure_reduction:
        result["artifacts"].update(write_failure_reduction(result, suite.output_dir))
    if suite.engineering.enabled:
        result["artifacts"].update(write_suite_engineering_exports(result, suite.output_dir))
    result["artifacts"].update(write_controller_bench_reports(result, suite.output_dir))
    return result

__all__ = [name for name in globals() if not name.startswith("__")]
