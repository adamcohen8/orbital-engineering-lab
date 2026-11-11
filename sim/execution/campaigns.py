# ruff: noqa: F401,F821,I001
"""Compatibility façade for config and Monte Carlo campaign execution."""

from . import campaign_common as _common
from . import config_campaigns as _config
from . import monte_carlo_preparation as _preparation
from . import monte_carlo_checkpoints as _checkpoints
from . import monte_carlo_campaign_execution as _execution
from . import monte_carlo_review as _review
from . import monte_carlo_aggregation as _aggregation
from sim.execution.thread_campaign import (
    ThreadCampaignBackendUnavailable,
    ensure_thread_campaign_compatible,
)

CAMPAIGN_CAPABILITY_FAMILIES = {
    "config_campaigns": "sim.execution.config_campaigns",
    "checkpointing": "sim.execution.monte_carlo_checkpoints",
    "preparation": "sim.execution.monte_carlo_preparation",
    "execution": "sim.execution.monte_carlo_campaign_execution",
    "aggregation": "sim.execution.monte_carlo_aggregation",
    "review": "sim.execution.monte_carlo_review",
}

for _module in (_common, _config, _preparation, _checkpoints, _execution, _review, _aggregation):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})

ConfigCampaignRunSpec.__module__ = __name__
ConfigCampaignSpec.__module__ = __name__
del _module

def run_config_campaign(
    campaign_path: str | Path,
    *,
    import_plugins: bool = True,
    batch_callback: BatchCallback | None = None,
    batch_progress_callback: BatchProgressCallback | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Run a bounded local config-queue campaign of existing scenario YAML files."""

    require_pro_feature(FEATURE_CAMPAIGNS)
    campaign_source = Path(campaign_path).expanduser().resolve()
    campaign_content = campaign_source.read_bytes()
    campaign_sha256 = hashlib.sha256(campaign_content).hexdigest()
    campaign = load_config_campaign(campaign_path, _content=campaign_content)
    if hashlib.sha256(campaign.config_path.read_bytes()).hexdigest() != campaign_sha256:
        raise ValueError("campaign file changed while it was being loaded; retry with a stable file")
    loaded_configs: dict[Path, tuple[str, SimulationScenarioConfig]] = {}
    retention_policy = dict(campaign.outputs.get("campaign_retention", {}) or {})
    validate_first = bool(campaign.execution.get("validate_first", True))
    if validate_first or retention_policy:
        validation = validate_config_campaign(
            campaign.config_path,
            import_plugins=import_plugins if validate_first else False,
            _campaign=campaign,
            _campaign_sha256=campaign_sha256,
            _loaded_configs=loaded_configs,
        )
        if not bool(validation.get("valid", False)):
            errors = list(validation.get("errors", []) or [])
            raise ValueError("Campaign validation failed:\n- " + "\n- ".join(str(err) for err in errors))

    include_payloads = bool(campaign.outputs.get("include_payloads", False))
    tasks = []
    for spec in campaign.runs:
        cached = loaded_configs.get(spec.config_path)
        if cached is not None:
            expected_config_sha256 = cached[0]
        else:
            try:
                expected_config_sha256 = hashlib.sha256(spec.config_path.read_bytes()).hexdigest()
            except OSError:
                # Preserve validation-disabled campaign construction so the
                # worker can report a per-run failure without ever executing.
                expected_config_sha256 = ""
        tasks.append(
            _campaign_run_task(
                spec,
                include_payload=include_payloads,
                expected_config_sha256=expected_config_sha256,
                retention_policy=retention_policy,
                config_dict=(cached[1].to_dict() if cached is not None else None),
            )
        )
    mode = str(campaign.execution.get("mode", "serial") or "serial")
    stop_on_failure = bool(campaign.execution.get("stop_on_failure", True))
    requested_backend = backend if backend is not None else campaign.execution.get("backend", "process_pool")
    parallel_backend = str(requested_backend or "process_pool").strip().lower()
    if parallel_backend not in {"process_pool", "thread_pool"}:
        raise ValueError("Config campaign parallel backend must be 'process_pool' or 'thread_pool'.")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    parallel_workers = 1
    hierarchy: dict[str, Any] | None = None
    requested_mode = mode
    parallel_fallback_reason: str | None = None

    if mode == "parallel" and stop_on_failure and len(tasks) > 1:
        logger.info("Parallel config campaign requested with stop_on_failure=true; running serial for fail-fast behavior.")
        parallel_fallback_reason = "stop_on_failure=true requires serial fail-fast execution"
        mode = "serial"

    if mode == "parallel" and len(tasks) > 1:
        requested = int(campaign.execution.get("max_workers", 0) or 0)
        default_workers = max(1, (os.cpu_count() or 1) - 1)
        task_roots = []
        for spec in campaign.runs:
            try:
                cached = loaded_configs.get(spec.config_path)
                current_sha256 = hashlib.sha256(spec.config_path.read_bytes()).hexdigest()
                if cached is not None and cached[0] == current_sha256:
                    task_roots.append(cached[1].to_dict())
                elif cached is not None:
                    raise RuntimeError(
                        f"Campaign scenario changed after validation: {spec.config_path}"
                    )
                else:
                    task_roots.append(load_simulation_yaml(spec.config_path).to_dict())
            except (FileNotFoundError, OSError, ValueError):
                task_roots.append({})
        hierarchy_plan = plan_hierarchical_execution(
            task_roots=task_roots,
            task_count=len(tasks),
            requested_campaign_workers=requested if requested > 0 else default_workers,
            profile=restrictive_profile_for_task_roots(task_roots),
        )
        hierarchy = hierarchy_plan.payload()
        parallel_workers = int(hierarchy_plan.campaign_workers)
        if parallel_workers <= 1:
            parallel_fallback_reason = f"hierarchical_planner:{hierarchy_plan.reason}"
            mode = "serial"

    if mode == "parallel" and parallel_backend == "thread_pool" and len(tasks) > 1:
        for task in tasks:
            config_dict = task.get("config_dict")
            if not isinstance(config_dict, dict):
                if not validate_first:
                    raise ThreadCampaignBackendUnavailable(
                        "Thread config campaigns require execution.validate_first=true so "
                        "scenario extensions can be checked before worker output is created."
                    )
                config_path = str(dict(task.get("spec", {}) or {}).get("config_path", ""))
                config_dict = load_simulation_yaml(config_path).to_dict()
                task["config_dict"] = config_dict
            ensure_thread_campaign_compatible(
                config_dict,
                label=f"config campaign run {dict(task.get('spec', {}) or {}).get('name', '')!r}",
            )
            task["worker_budget"] = dict(hierarchy_plan.payload())
            task["progress_emit_every"] = int(campaign.execution.get("parallel_progress_emit_every_steps", 20) or 20)
        executor = ThreadCampaignExecutor(max_workers=parallel_workers)
        try:
            for future, _task in executor.iter_completed(_run_config_campaign_worker, tasks):
                if future is not None:
                    result = dict(future.result())
                    results.append(result)
                    if batch_callback is not None:
                        try:
                            batch_callback(len(results), len(tasks))
                        except BaseException as exc:
                            try:
                                _raise_if_cancellation(exc)
                            except BaseException:
                                executor.cancel()
                                raise
                            if not isinstance(exc, Exception):
                                executor.cancel()
                                raise
                            logger.warning("Disabling config campaign callback after runtime error: %s", exc)
                            batch_callback = None
                _unused_step_callback, batch_progress_callback = _drain_parallel_progress_events(
                    executor.progress_queue,
                    step_callback=None,
                    batch_progress_callback=batch_progress_callback,
                    callback_label="threaded config campaign",
                )
        except BaseException:
            executor.cancel()
            raise
        finally:
            try:
                _drain_parallel_progress_events(
                    executor.progress_queue,
                    step_callback=None,
                    batch_progress_callback=batch_progress_callback,
                    callback_label="threaded config campaign",
                )
            finally:
                executor.shutdown(wait_for_running=True)

    if mode == "parallel" and parallel_backend == "process_pool" and len(tasks) > 1:
        manager = None
        progress_queue = None
        parallel_error: BaseException | None = None
        thread_env_prev = set_parallel_worker_thread_limits(default_threads="1")
        hierarchy_env_prev = apply_hierarchical_worker_env(hierarchy_plan)
        try:
            if batch_progress_callback is not None:
                try:
                    manager = mp.Manager()
                    progress_queue = manager.Queue()
                except (OSError, PermissionError, EOFError) as exc:
                    logger.warning(
                        "Parallel config campaign progress transport is unavailable; "
                        "continuing without live progress: %s",
                        exc,
                    )
                    manager = None
                    progress_queue = None
            for task in tasks:
                task["progress_emit_every"] = int(campaign.execution.get("parallel_progress_emit_every_steps", 20) or 20)
            pool_kwargs = {}
            if progress_queue is not None:
                pool_kwargs = {
                    "initializer": initialize_worker_progress_queue,
                    "initargs": (progress_queue,),
                }
            with ProcessPoolExecutor(max_workers=parallel_workers, **pool_kwargs) as executor:
                for future, _task in iter_bounded_futures(
                    executor,
                    _run_config_campaign_worker,
                    tasks,
                    max_in_flight=2 * parallel_workers,
                ):
                    if future is not None:
                        try:
                            result = dict(future.result())
                        except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
                            parallel_error = exc
                            abort_process_pool(executor)
                            break
                        except BaseException:
                            abort_process_pool(executor)
                            raise
                        else:
                            results.append(result)
                            if batch_callback is not None:
                                try:
                                    batch_callback(len(results), len(tasks))
                                except BaseException as exc:
                                    try:
                                        _raise_if_cancellation(exc)
                                    except BaseException:
                                        abort_process_pool(executor)
                                        raise
                                    if not isinstance(exc, Exception):
                                        abort_process_pool(executor)
                                        raise
                                    logger.warning("Disabling config campaign callback after runtime error: %s", exc)
                                    batch_callback = None
                    if progress_queue is not None:
                        while True:
                            try:
                                evt = progress_queue.get_nowait()
                            except queue_mod.Empty:
                                break
                            except Exception:
                                break
                            if batch_progress_callback is not None:
                                try:
                                    batch_progress_callback(dict(evt or {}))
                                except BaseException as exc:
                                    try:
                                        _raise_if_cancellation(exc)
                                    except BaseException:
                                        abort_process_pool(executor)
                                        raise
                                    if not isinstance(exc, Exception):
                                        abort_process_pool(executor)
                                        raise
                                    logger.warning(
                                        "Disabling config campaign progress callback after runtime error: %s", exc
                                    )
                                    batch_progress_callback = None
                if parallel_error is not None:
                    raise parallel_error
        except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
            logger.warning("Parallel config campaign unavailable, falling back to serial execution: %s", exc)
            parallel_fallback_reason = format_parallel_fallback_reason(exc)
            mode = "serial"
        finally:
            if progress_queue is not None:
                while True:
                    try:
                        evt = progress_queue.get_nowait()
                    except queue_mod.Empty:
                        break
                    except Exception:
                        break
                    if batch_progress_callback is not None:
                        try:
                            batch_progress_callback(dict(evt or {}))
                        except Exception as exc:
                            _raise_if_cancellation(exc)
                            logger.warning(
                                "Disabling config campaign progress callback after runtime error: %s", exc
                            )
                            batch_progress_callback = None
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            restore_env_vars(thread_env_prev)
            restore_hierarchical_worker_env(hierarchy_env_prev)
    if mode != "parallel" or len(tasks) <= 1:
        mode = "serial"
        completed_indices = {int(item.get("index", -1)) for item in results}
        for task in tasks:
            index = int(dict(task.get("spec", {}) or {}).get("index", -1))
            if index in completed_indices:
                continue
            task = dict(task)
            task["progress_queue"] = None
            result = _run_config_campaign_worker(task)
            results.append(result)
            if batch_callback is not None:
                try:
                    batch_callback(len(results), len(tasks))
                except Exception as exc:
                    _raise_if_cancellation(exc)
                    logger.warning("Disabling config campaign callback after runtime error: %s", exc)
                    batch_callback = None
            if stop_on_failure and str(result.get("status", "")) != "passed":
                break

    results.sort(key=lambda item: int(item.get("index", 0)))
    dataset_artifacts: dict[str, str] = {}
    if retention_policy:
        from .campaign_retention import write_campaign_dataset

        dataset_records = []
        for result in results:
            dataset_records.append({
                "iteration": int(result.get("index", 0)),
                "name": str(result.get("name", "")),
                "config_path": str(result.get("config_path", "")),
                "config_sha256": str(tasks[int(result.get("index", 0))]["spec"].get("expected_config_sha256", "")),
                "metadata": dict(result.get("metadata", {}) or {}),
                "status": str(result.get("status", "")),
                "error": result.get("error"),
                "summary": dict(result.get("summary", {}) or {}),
                "retained_evidence": dict(result.pop("retained_evidence", {}) or {}),
                "full_review": result.get("full_review"),
            })
        dataset_artifacts = write_campaign_dataset(
            campaign.outputs["root"], policy=retention_policy, records=dataset_records
        )
    failed = [item for item in results if str(item.get("status", "")) != "passed"]
    payload = {
        "campaign": {
            "name": campaign.name,
            "description": campaign.description,
            "config_path": str(campaign.config_path),
        },
        "execution": {
            "mode": mode,
            "requested_mode": requested_mode,
            "parallel_backend": parallel_backend if mode == "parallel" else "serial",
            "parallel_workers": int(parallel_workers),
            "parallel_fallback_reason": parallel_fallback_reason,
            "hierarchical_execution": hierarchy,
            "validate_first": bool(campaign.execution.get("validate_first", True)),
            "stop_on_failure": stop_on_failure,
        },
        "summary": {
            "run_count": len(campaign.runs),
            "executed_count": len(results),
            "passed_count": len(results) - len(failed),
            "failed_count": len(failed),
            "wall_time_s": float(time.perf_counter() - started),
        },
        "runs": results,
        "artifacts": {},
    }
    artifacts = _write_config_campaign_artifacts(campaign, payload)
    artifacts.update(dataset_artifacts)
    payload["artifacts"] = artifacts
    write_json(artifacts["summary_json"], payload)
    return payload
