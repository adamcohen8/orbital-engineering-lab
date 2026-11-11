# ruff: noqa: F401,F403,F405,I001
import hashlib
import json
import threading
from dataclasses import asdict

from sim.utils.parallel import abort_process_pool
from sim.execution.thread_campaign import ThreadCampaignExecutor, ensure_thread_campaign_compatible

from .bench_common import *
from .config import *
from .variants import *
from .cases import *
from .criteria import *
from .diagnostics import *
from .objectives import *

def _run_single_bench_case_worker(task: dict[str, Any]) -> dict[str, Any]:
    index = int(task.get("index", 0))
    suite = task["suite"]
    case = task["case"]
    variant = task["variant"]
    progress_queue = task.get("progress_queue")
    cancel_event = task.get("cancel_event")
    emit_every = max(1, int(task.get("progress_emit_every", 20) or 20))
    last_emit = -(10**9)

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        if cancel_event is not None and cancel_event.is_set():
            from sim.execution.thread_campaign import ThreadCampaignCancelled

            raise ThreadCampaignCancelled("Thread controller-bench worker cancelled.")
        if progress_queue is None:
            return
        s = max(int(step), 0)
        t = max(int(total), 0)
        if s != 0 and not (t > 0 and s >= t) and s - last_emit < emit_every:
            return
        last_emit = s
        try:
            progress_queue.put(
                {
                    "event": "step",
                    "pid": int(os.getpid()),
                    "thread_id": int(threading.get_ident()),
                    "iteration": index,
                    "case_name": case.name,
                    "variant_name": variant.name,
                    "step": s,
                    "total": t,
                }
            )
        except Exception:
            pass

    step_callback = _on_step if progress_queue is not None or cancel_event is not None else None
    run, _payload = _run_single_bench_case_with_payload(
        suite=suite,
        case=case,
        variant=variant,
        step_callback=step_callback,
    )
    return {
        "index": index,
        "run": run,
    }


def _deep_get_value(root: dict[str, Any], path: str) -> Any:
    cur: Any = root
    for token in [tok for tok in str(path).split(".") if tok]:
        if isinstance(cur, dict) and token in cur:
            cur = cur[token]
            continue
        if isinstance(cur, list) and token.isdigit():
            idx = int(token)
            if 0 <= idx < len(cur):
                cur = cur[idx]
                continue
        raise KeyError(path)
    return cur


def _apply_optimization_vector(
    pointer: dict[str, Any],
    parameters: tuple[ControllerBenchOptimizationParameter, ...],
    x: np.ndarray,
) -> dict[str, Any]:
    out = deepcopy(pointer)
    flat = np.array(x, dtype=float).reshape(-1)
    expected_dim = sum(_optimization_parameter_dim(spec) for spec in parameters)
    if expected_dim != int(flat.size):
        raise ValueError("Optimization vector length does not match parameter specification.")
    cursor = 0
    for spec in parameters:
        dim = _optimization_parameter_dim(spec)
        values = flat[cursor : cursor + dim]
        cursor += dim
        root_path = spec.path
        if root_path.startswith("controller."):
            root_path = root_path[len("controller.") :]
        if dim == 1 and not isinstance(spec.lower, tuple):
            _deep_set(out, root_path, float(values[0]))
        else:
            _deep_set(out, root_path, _unflatten_numeric_tree(values.tolist(), spec.lower))
    return out


def _optimized_variant_from_vector(
    base_variant: ControllerVariant,
    parameters: tuple[ControllerBenchOptimizationParameter, ...],
    x: np.ndarray,
    *,
    name: str,
    description: str,
) -> ControllerVariant:
    return ControllerVariant(
        name=name,
        pointer=_apply_optimization_vector(base_variant.pointer, parameters, x),
        description=description,
        pointers=deepcopy(base_variant.pointers),
        overrides=deepcopy(base_variant.overrides),
    )


def _flatten_numeric_tree(value: Any) -> list[float]:
    if isinstance(value, tuple):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric_tree(item))
        return out
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric_tree(item))
        return out
    return [float(value)]


def _unflatten_numeric_tree(values: list[float], template: Any) -> Any:
    cursor = 0

    def _build(node: Any) -> Any:
        nonlocal cursor
        if isinstance(node, tuple):
            return [_build(item) for item in node]
        value = float(values[cursor])
        cursor += 1
        return value

    return _build(template)


def _optimization_parameter_dim(spec: ControllerBenchOptimizationParameter) -> int:
    return len(_flatten_numeric_tree(spec.lower))


def _optimization_bounds(
    parameters: tuple[ControllerBenchOptimizationParameter, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    for spec in parameters:
        spec_lower = _flatten_numeric_tree(spec.lower)
        spec_upper = _flatten_numeric_tree(spec.upper)
        if len(spec_lower) != len(spec_upper):
            raise ValueError(f"optimization parameter '{spec.name}' lower/upper bounds must have matching shape.")
        lower.extend(spec_lower)
        upper.extend(spec_upper)
        if len(spec_lower) == 1:
            names.append(spec.name)
        else:
            names.extend(f"{spec.name}[{idx}]" for idx in range(len(spec_lower)))
    return np.array(lower, dtype=float), np.array(upper, dtype=float), names


def _optimization_parameter_values(
    parameters: tuple[ControllerBenchOptimizationParameter, ...],
    x: np.ndarray,
) -> dict[str, Any]:
    flat = np.array(x, dtype=float).reshape(-1)
    out: dict[str, Any] = {}
    cursor = 0
    for spec in parameters:
        dim = _optimization_parameter_dim(spec)
        values = flat[cursor : cursor + dim]
        cursor += dim
        if dim == 1 and not isinstance(spec.lower, tuple):
            out[spec.name] = float(values[0])
        else:
            out[spec.name] = _unflatten_numeric_tree(values.tolist(), spec.lower)
    return out


def _optimization_seed_vector(
    pointer: dict[str, Any],
    parameters: tuple[ControllerBenchOptimizationParameter, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray | None:
    values: list[float] = []
    for spec in parameters:
        root_path = spec.path
        if root_path.startswith("controller."):
            root_path = root_path[len("controller.") :]
        try:
            raw = _deep_get_value(pointer, root_path)
        except KeyError:
            return None
        flat = _flatten_numeric_tree(raw)
        if len(flat) != _optimization_parameter_dim(spec):
            return None
        values.extend(flat)
    seed = np.array(values, dtype=float).reshape(-1)
    if seed.shape != lower.shape:
        return None
    return np.clip(seed, lower, upper)


def _variant_report_entry(variant: ControllerVariant) -> dict[str, Any]:
    entry = {
        "name": variant.name,
        "description": variant.description,
        "pointer": deepcopy(variant.pointer),
        "linear_system_summary": _controller_linear_system_summary(variant.pointer),
    }
    if variant.pointers:
        entry["pointers"] = [
            {
                "object_id": pointer.object_id,
                "slot": pointer.slot,
                "pointer": deepcopy(pointer.pointer),
            }
            for pointer in variant.pointers
        ]
    if variant.overrides:
        entry["overrides"] = [
            {
                "path": override.path,
                "value": deepcopy(override.value),
            }
            for override in variant.overrides
        ]
    return entry


def _relative_cost_inputs(
    payload: dict[str, Any],
    cost: ControllerBenchOptimizationCost,
) -> dict[str, float]:
    truth_map = dict(payload.get("truth_by_object", {}) or {})
    obj = np.array(truth_map.get(cost.object_id, []), dtype=float)
    ref = np.array(truth_map.get(cost.reference_object_id, []), dtype=float)
    if obj.ndim != 2 or ref.ndim != 2 or obj.shape[0] == 0 or ref.shape[0] == 0:
        return {
            "terminal_relative_distance_km": float("inf"),
            "terminal_relative_speed_km_s": float("inf"),
            "fuel_used_kg": float("inf"),
            "time_to_rendezvous_s": float("inf"),
        }
    n = int(min(obj.shape[0], ref.shape[0]))
    dr = obj[:n, :3] - ref[:n, :3]
    dv = obj[:n, 3:6] - ref[:n, 3:6]
    dist = np.linalg.norm(dr, axis=1)
    speed = np.linalg.norm(dv, axis=1)

    t_s = np.array(payload.get("time_s", []), dtype=float).reshape(-1)
    if t_s.size >= n:
        time_hist = t_s[:n]
        duration_s = float(time_hist[-1]) if time_hist.size else 0.0
    else:
        time_hist = np.array([], dtype=float)
        summary = dict(payload.get("summary", {}) or {})
        duration_s = float(summary.get("duration_s", max(n - 1, 0)))

    inside = np.where(dist <= float(cost.rendezvous_radius_km))[0]
    if inside.size:
        first_idx = int(inside[0])
        time_to_rendezvous_s = float(time_hist[first_idx]) if time_hist.size > first_idx else float(first_idx)
    else:
        time_to_rendezvous_s = float(duration_s + cost.miss_time_penalty_s)

    fuel_used_kg = 0.0
    if obj.shape[1] >= 14:
        mass = obj[:n, 13]
        finite = mass[np.isfinite(mass)]
        if finite.size:
            fuel_used_kg = float(max(0.0, finite[0] - finite[-1]))
    else:
        summary = dict(payload.get("summary", {}) or {})
        thrust_stats = dict(summary.get("thrust_stats", {}) or {})
        fuel_used_kg = float(dict(thrust_stats.get(cost.object_id, {}) or {}).get("fuel_used_kg", 0.0) or 0.0)

    return {
        "terminal_relative_distance_km": float(dist[-1]),
        "terminal_relative_speed_km_s": float(speed[-1]),
        "fuel_used_kg": fuel_used_kg,
        "time_to_rendezvous_s": time_to_rendezvous_s,
    }


def _optimization_cost_from_payload(
    payload: dict[str, Any],
    cost: ControllerBenchOptimizationCost,
) -> tuple[float, dict[str, float]]:
    vals = _relative_cost_inputs(payload, cost)
    pos_scale = max(float(cost.terminal_position_scale_km), 1e-12)
    speed_scale = max(float(cost.terminal_speed_scale_km_s), 1e-12)
    fuel_scale = max(float(cost.fuel_scale_kg), 1e-12)
    time_scale = max(float(cost.time_scale_s), 1e-12)
    terminal_state_cost = math.hypot(
        float(vals["terminal_relative_distance_km"]) / pos_scale,
        float(vals["terminal_relative_speed_km_s"]) / speed_scale,
    )
    fuel_cost = float(vals["fuel_used_kg"]) / fuel_scale
    time_cost = float(vals["time_to_rendezvous_s"]) / time_scale
    total = (
        float(cost.terminal_state_weight) * terminal_state_cost
        + float(cost.fuel_weight) * fuel_cost
        + float(cost.time_weight) * time_cost
    )
    vals.update(
        {
            "terminal_state_cost": terminal_state_cost,
            "fuel_cost": fuel_cost,
            "time_cost": time_cost,
            "total_cost": float(total),
        }
    )
    return float(total), vals


def _criterion_normalized_cost(value: Any, criterion: ControllerBenchPassCriterion) -> tuple[float, dict[str, Any]]:
    value_f = _safe_metric_float(value)
    threshold_f = _safe_metric_float(criterion.value)
    ok, reason, margin = _evaluate_single_pass_criterion(value, criterion)
    detail = {
        "metric": criterion.metric,
        "op": criterion.op,
        "threshold": criterion.value,
        "value": value,
        "passed": bool(ok),
        "reason": reason,
        "margin": margin,
    }
    if value_f is None or threshold_f is None or not math.isfinite(value_f) or not math.isfinite(threshold_f):
        return 1000000.0, detail
    scale = max(abs(float(threshold_f)), 1.0)
    if criterion.op in {"<=", "<"}:
        normalized = max(0.0, float(value_f) / scale)
        violation = max(0.0, float(value_f) - float(threshold_f)) / scale
    elif criterion.op in {">=", ">"}:
        normalized = max(0.0, (float(threshold_f) - float(value_f)) / scale)
        violation = max(0.0, float(threshold_f) - float(value_f)) / scale
    else:
        return (0.0 if ok else 1000.0), detail
    cost = normalized
    if not ok:
        cost += 1000.0 + 100.0 * violation
    return float(cost), detail


def _generic_optimization_cost_from_run(
    run: dict[str, Any],
    criteria: tuple[ControllerBenchPassCriterion, ...],
) -> tuple[float, dict[str, Any]]:
    metrics = dict(run.get("metrics", {}) or {})
    criterion_details: list[dict[str, Any]] = []
    total = 0.0
    for criterion in criteria:
        cost, detail = _criterion_normalized_cost(metrics.get(criterion.metric), criterion)
        total += cost
        criterion_details.append(detail)
    if not criteria:
        total += 0.0 if bool(run.get("passed", False)) else 1000.0
    failure_analysis = dict(run.get("failure_analysis", {}) or {})
    failure_mode_counts = dict(failure_analysis.get("failure_mode_counts", {}) or {})
    failures = list(failure_analysis.get("failures", []) or [])
    total += 250.0 * len(failures)
    for name, value in metrics.items():
        metric_name = str(name)
        if not metric_name.endswith(("total_dv_m_s", "fuel_used_kg", "total_torque_norm_n_m_s")):
            continue
        value_f = _safe_metric_float(value)
        if value_f is not None and math.isfinite(value_f):
            total += 0.01 * max(0.0, float(value_f))
    return float(total), {
        "cost_mode": "generic_objective_score",
        "total_cost": float(total),
        "passed": bool(run.get("passed", False)),
        "failure_count": int(len(failures)),
        "failure_mode_counts": failure_mode_counts,
        "criteria": criterion_details,
    }


def _evaluate_particle_worker(task: dict[str, Any]) -> dict[str, Any]:
    suite: ControllerBenchConfig = task["suite"]
    cases: list[ControllerBenchCase] = list(task["cases"])
    base_variant: ControllerVariant = task["base_variant"]
    x = np.array(task["x"], dtype=float).reshape(-1)
    iteration = int(task.get("iteration", 0))
    particle = int(task.get("particle", 0))
    progress_queue = task.get("progress_queue")
    if "progress_queue" not in task:
        progress_queue = worker_progress_queue()
    emit_every = max(1, int(task.get("progress_emit_every", 20) or 20))
    variant = _optimized_variant_from_vector(
        base_variant,
        suite.optimization.parameters,
        x,
        name=f"{base_variant.name}__pso_i{iteration:03d}_p{particle:03d}",
        description=f"PSO particle {particle} iteration {iteration}",
    )
    runs: list[dict[str, Any]] = []
    case_costs: list[float] = []
    case_cost_details: list[dict[str, Any]] = []

    def _make_on_step(case_idx: int, case_name: str):
        last_emit = -(10**9)

        def _on_step(step: int, total: int) -> None:
            nonlocal last_emit
            cancel_event = task.get("cancel_event")
            if cancel_event is not None and cancel_event.is_set():
                from sim.execution.thread_campaign import ThreadCampaignCancelled

                raise ThreadCampaignCancelled("Thread controller-bench optimization worker cancelled.")
            if progress_queue is None:
                return
            s = max(int(step), 0)
            t = max(int(total), 0)
            should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
            if not should_emit:
                return
            last_emit = s
            try:
                progress_queue.put(
                    {
                        "event": "step",
                        "pid": int(os.getpid()),
                        "iteration": int(iteration),
                        "particle": int(particle),
                        "case_index": int(case_idx),
                        "case_name": case_name,
                        "step": int(s),
                        "total": int(t),
                    }
                )
            except Exception:
                pass

        return _on_step

    for case_idx, case in enumerate(cases):
        step_callback = _make_on_step(case_idx, case.name) if progress_queue is not None else None
        run, payload = _run_single_bench_case_with_payload(
            suite=suite,
            case=case,
            variant=variant,
            step_callback=step_callback,
        )
        if suite.optimization.cost is not None:
            cost_value, cost_details = _optimization_cost_from_payload(payload, suite.optimization.cost)
        else:
            objectives = case.objectives if case.objectives else suite.objectives
            criteria = _resolve_pass_criteria(suite, case, objectives)
            cost_value, cost_details = _generic_optimization_cost_from_run(run, criteria)
        run["optimization_cost"] = cost_value
        run["optimization_cost_details"] = cost_details
        runs.append(run)
        if math.isfinite(cost_value):
            case_costs.append(float(cost_value))
        else:
            case_costs.append(float("inf"))
        case_cost_details.append({"case_name": case.name, **cost_details})
    if progress_queue is not None:
        try:
            progress_queue.put(
                {
                    "event": "done",
                    "pid": int(os.getpid()),
                    "iteration": int(iteration),
                    "particle": int(particle),
                }
            )
        except Exception:
            pass
    mean_cost = float(np.mean(np.array(case_costs, dtype=float))) if case_costs else float("inf")
    return {
        "iteration": iteration,
        "particle": particle,
        "x": x.tolist(),
        "parameters": _optimization_parameter_values(suite.optimization.parameters, x),
        "cost": mean_cost,
        "case_cost_details": case_cost_details,
        "runs": runs,
    }


def _evaluate_particle_swarm(
    *,
    suite: ControllerBenchConfig,
    cases: list[ControllerBenchCase],
    base_variant: ControllerVariant,
    swarm: np.ndarray,
    iteration: int,
    workers: int,
    particle_done_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_event_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_emit_every: int = 20,
    execution_meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    task_roots = [load_simulation_yaml(case.config_path).to_dict() for case in cases]
    hierarchy_plan = plan_hierarchical_execution(
        task_roots=task_roots,
        task_count=int(swarm.shape[0]),
        requested_campaign_workers=workers,
        profile=restrictive_profile_for_task_roots(task_roots),
    )
    workers = int(hierarchy_plan.campaign_workers)
    manager = None
    progress_queue = None
    # A process manager is part of the legacy process transport.  Thread
    # execution receives its queue from ThreadCampaignExecutor and must not
    # create a manager process as a side effect of opting into threads.
    if suite.parallel_backend == "process_pool" and progress_event_callback is not None:
        try:
            manager = mp.Manager()
            progress_queue = manager.Queue()
        except (OSError, PermissionError, EOFError) as exc:
            logger.warning(
                "Parallel optimization progress transport is unavailable; continuing without live progress: %s",
                exc,
            )
            manager = None
            progress_queue = None
    tasks = [
        {
            "suite": suite,
            "cases": cases,
            "base_variant": base_variant,
            "x": swarm[idx, :].tolist(),
            "iteration": iteration,
            "particle": idx,
            "progress_emit_every": int(progress_emit_every),
            "worker_budget": dict(hierarchy_plan.payload()),
        }
        for idx in range(swarm.shape[0])
    ]
    out_by_particle: dict[int, dict[str, Any]] = {}
    if suite.parallel_backend == "thread_pool":
        for case in cases:
            ensure_thread_campaign_compatible(
                load_simulation_yaml(case.config_path).to_dict(),
                label=f"controller optimization case {case.name!r}",
            )
        executor = ThreadCampaignExecutor(max_workers=workers)
        try:
            for future, task in executor.iter_completed(_evaluate_particle_worker, tasks):
                if future is not None:
                    item = future.result()
                    particle_idx = int(item.get("particle", task["particle"]))
                    out_by_particle[particle_idx] = item
                    if particle_done_callback is not None:
                        particle_done_callback(item)
                if progress_event_callback is not None:
                    for event in executor.drain_progress():
                        progress_event_callback(dict(event))
        except BaseException:
            executor.cancel()
            raise
        finally:
            try:
                if progress_event_callback is not None:
                    for event in executor.drain_progress():
                        progress_event_callback(dict(event))
            finally:
                executor.shutdown(wait_for_running=True)
        return [out_by_particle[idx] for idx in sorted(out_by_particle.keys())]

    hierarchy_env_prev = apply_hierarchical_worker_env(hierarchy_plan)
    parallel_error: BaseException | None = None
    ex = None
    try:
        pool_kwargs = {}
        if progress_queue is not None:
            pool_kwargs = {
                "initializer": initialize_worker_progress_queue,
                "initargs": (progress_queue,),
            }
        ex = ProcessPoolExecutor(max_workers=workers, **pool_kwargs)
        try:
            for fut, task in iter_bounded_futures(
                ex,
                _evaluate_particle_worker,
                tasks,
                max_in_flight=2 * workers,
            ):
                if fut is not None:
                    try:
                        item = fut.result()
                    except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
                        parallel_error = exc
                        abort_process_pool(ex)
                        break
                    except BaseException:
                        abort_process_pool(ex)
                        raise
                    else:
                        particle_idx = int(item.get("particle", task["particle"]))
                        out_by_particle[particle_idx] = item
                        if particle_done_callback is not None:
                            particle_done_callback(item)
                if progress_queue is not None and progress_event_callback is not None:
                    while True:
                        try:
                            evt = progress_queue.get_nowait()
                        except queue_mod.Empty:
                            break
                        except Exception:
                            break
                    progress_event_callback(dict(evt or {}))
            if parallel_error is not None:
                raise parallel_error
        finally:
            shutdown = getattr(ex, "shutdown", None)
            if ex is not None and parallel_error is None and callable(shutdown):
                shutdown(wait=True, cancel_futures=False)
    except (OSError, PermissionError, NotImplementedError, EOFError, BrokenProcessPool) as exc:
        reason = format_parallel_fallback_reason(exc)
        logger.warning("Parallel PSO evaluation unavailable, finishing missing particles serially: %s", exc)
        if execution_meta is not None:
            execution_meta["parallel_fallback_reason"] = reason
        for task in tasks:
            particle_idx = int(task["particle"])
            if particle_idx in out_by_particle:
                continue
            serial_task = dict(task)
            serial_task["progress_queue"] = None
            item = _evaluate_particle_worker(serial_task)
            out_by_particle[particle_idx] = item
            if particle_done_callback is not None:
                particle_done_callback(item)
    finally:
        restore_hierarchical_worker_env(hierarchy_env_prev)
        if progress_queue is not None and progress_event_callback is not None:
            try:
                while True:
                    evt = progress_queue.get_nowait()
                    progress_event_callback(dict(evt or {}))
            except Exception:
                pass
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
    return [out_by_particle[idx] for idx in sorted(out_by_particle.keys())]


def _available_pso_worker_cap() -> int:
    return max(1, int(os.cpu_count() or 1) - 1)


def _pso_contract_digest(
    *,
    suite: ControllerBenchConfig,
    base_variant: ControllerVariant,
    source_config_path: str,
    particles: int,
) -> str:
    case_inputs = []
    for case in suite.cases:
        path = Path(case.config_path)
        case_inputs.append(
            {
                "name": case.name,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    source = Path(source_config_path) if source_config_path else None
    implementation_paths = tuple(
        Path(__file__).with_name(name)
        for name in (
            "config.py",
            "variants.py",
            "cases.py",
            "criteria.py",
            "metrics.py",
            "objectives.py",
            "diagnostics.py",
            "engineering.py",
            "optimization.py",
            "reporting.py",
            "runner.py",
        )
    )
    implementation_paths += tuple(
        Path(__file__).resolve().parents[1] / path
        for path in (
            "flight_software/reference_stacks.py",
            "runtime/satellites/flight_software_runtime.py",
            "runtime/satellites/factory.py",
            "actuators/command_bus.py",
            "actuators/physical.py",
            "dynamics/orbit/propagator.py",
            "dynamics/orbit/accelerations.py",
            "rocket/guidance.py",
            "rocket/engine.py",
        )
    )
    payload = {
        "suite_name": suite.suite_name,
        "source_config_path": str(source or ""),
        "source_config_sha256": (
            hashlib.sha256(source.read_bytes()).hexdigest() if source is not None and source.is_file() else ""
        ),
        "case_inputs": case_inputs,
        "implementation_dependencies": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in implementation_paths
            if path.is_file()
        ],
        "base_variant": _variant_report_entry(base_variant),
        "controller_target": asdict(suite.controller_target),
        "suite_metrics": [asdict(item) for item in suite.metrics],
        "suite_pass_criteria": [asdict(item) for item in suite.pass_criteria],
        "suite_objectives": [asdict(item) for item in suite.objectives],
        "case_contracts": [
            {
                "name": case.name,
                "metrics": [asdict(item) for item in case.metrics],
                "pass_criteria": [asdict(item) for item in case.pass_criteria],
                "objectives": [asdict(item) for item in case.objectives],
            }
            for case in suite.cases
        ],
        "optimization": {
            "algorithm": suite.optimization.algorithm,
            "particles": particles,
            "iterations": suite.optimization.iterations,
            "inertia": suite.optimization.inertia,
            "cognitive": suite.optimization.cognitive,
            "social": suite.optimization.social,
            "seed": suite.optimization.seed,
            "base_variant": suite.optimization.base_variant,
            "cost": None if suite.optimization.cost is None else asdict(suite.optimization.cost),
            "parameters": [
                {
                    "name": item.name,
                    "path": item.path,
                    "lower": item.lower,
                    "upper": item.upper,
                }
                for item in suite.optimization.parameters
            ],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_pso_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_pso_checkpoint(path: Path, *, contract_digest: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("contract_digest") != contract_digest:
        return None
    return payload


def _run_controller_bench_optimization(
    suite: ControllerBenchConfig,
    *,
    compare_names: list[str] | None = None,
    source_config_path: str = "",
    show_progress: bool = True,
) -> dict[str, Any]:
    opt = suite.optimization
    if compare_names:
        raise ValueError("Controller bench optimization cannot be combined with --compare.")
    if opt.particles <= 0 or opt.iterations <= 0:
        raise ValueError("optimization.particles and optimization.iterations must be positive.")
    if opt.inertia < 0.0 or opt.cognitive < 0.0 or opt.social < 0.0:
        raise ValueError("PSO inertia, cognitive, and social terms must be non-negative.")

    variants = list(suite.variants)
    if compare_names:
        wanted = {str(name) for name in compare_names}
        variants = [variant for variant in variants if variant.name in wanted]
    base_variant = None
    if opt.base_variant:
        for variant in suite.variants:
            if variant.name == opt.base_variant:
                base_variant = variant
                break
        if base_variant is None:
            raise ValueError(f"optimization.base_variant '{opt.base_variant}' does not match a configured variant.")
    elif variants:
        base_variant = variants[0]
    if base_variant is None:
        raise ValueError("Controller bench optimization requires at least one base variant.")

    rng = np.random.default_rng(int(opt.seed))
    lb, ub, parameter_vector_names = _optimization_bounds(opt.parameters)
    dim = int(lb.size)
    span = ub - lb
    requested_particles = int(opt.particles)
    worker_cap = _available_pso_worker_cap()
    particles = requested_particles
    workers = min(particles, worker_cap)
    optimization_task_roots = [load_simulation_yaml(case.config_path).to_dict() for case in suite.cases]
    optimization_hierarchy = plan_hierarchical_execution(
        task_roots=optimization_task_roots,
        task_count=particles,
        requested_campaign_workers=workers,
        profile=restrictive_profile_for_task_roots(optimization_task_roots),
    )
    workers = int(optimization_hierarchy.campaign_workers)
    x = lb + rng.random((particles, dim)) * span
    seed_x = _optimization_seed_vector(base_variant.pointer, opt.parameters, lb, ub)
    if seed_x is not None and particles > 0:
        x[0, :] = seed_x
    v = rng.uniform(-0.2, 0.2, size=(particles, dim)) * span

    suite.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = suite.output_dir / "controller_bench_pso_checkpoint.json"
    partial_path = suite.output_dir / "controller_bench_pso_partial_evaluations.json"
    contract_digest = _pso_contract_digest(
        suite=suite,
        base_variant=base_variant,
        source_config_path=source_config_path,
        particles=particles,
    )
    checkpoint_payload = _load_pso_checkpoint(checkpoint_path, contract_digest=contract_digest)
    resumed_from_checkpoint = checkpoint_payload is not None

    thread_env_prev = (
        set_parallel_worker_thread_limits(default_threads="1")
        if suite.parallel_backend == "process_pool"
        else {}
    )
    all_evaluations: list[dict[str, Any]] = (
        list(checkpoint_payload.get("evaluations", []) or []) if checkpoint_payload is not None else []
    )
    partial_evaluations: list[dict[str, Any]] = []
    optimization_execution_meta: dict[str, Any] = {}
    use_tqdm = bool(show_progress) and _should_use_tqdm()
    pso_bar = None
    worker_bars = []
    pid_to_slot: dict[int, int] = {}
    slot_state: dict[int, dict[str, Any]] = {}
    total_particle_evals = particles * (int(opt.iterations) + 1)
    completed_particle_evals = len(all_evaluations)
    plain_progress = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm  # type: ignore

            pso_bar = tqdm(total=total_particle_evals, desc="PSO", unit="particle", position=0)
            if completed_particle_evals:
                pso_bar.update(min(completed_particle_evals, total_particle_evals))
            for slot in range(workers):
                bar = tqdm(
                    total=1,
                    desc=f"Worker {slot + 1}",
                    unit="step",
                    position=slot + 1,
                    leave=False,
                    dynamic_ncols=True,
                )
                worker_bars.append(bar)
                slot_state[slot] = {"key": None, "last_step": 0}
        except Exception:
            pso_bar = None
            worker_bars = []
            plain_progress = _make_plain_progress_reporter("PSO")
    elif show_progress:
        plain_progress = _make_plain_progress_reporter("PSO")

    def _on_particle_done(item: dict[str, Any]) -> None:
        nonlocal completed_particle_evals, pso_bar, plain_progress
        completed_particle_evals += 1
        partial_evaluations.append(item)
        _write_pso_state(
            partial_path,
            {
                "schema_version": 1,
                "status": "partial",
                "contract_digest": contract_digest,
                "evaluations": partial_evaluations,
            },
        )
        if pso_bar is not None:
            pso_bar.update(1)
            try:
                pso_bar.set_postfix_str(f"best={float(item.get('cost', float('nan'))):.6g}")
            except Exception:
                pass
        elif plain_progress is not None:
            plain_progress(
                completed_particle_evals,
                total_particle_evals,
                f"particle {item.get('particle')} cost={item.get('cost')}",
            )

    def _on_worker_progress(evt: dict[str, Any]) -> None:
        nonlocal pid_to_slot, slot_state, worker_bars
        if not worker_bars:
            return
        event = str(evt.get("event", ""))
        pid = int(evt.get("pid", -1))
        if pid <= 0:
            return
        if pid not in pid_to_slot:
            used = set(pid_to_slot.values())
            free_slots = [idx for idx in range(len(worker_bars)) if idx not in used]
            pid_to_slot[pid] = free_slots[0] if free_slots else (len(pid_to_slot) % len(worker_bars))
        slot = int(pid_to_slot[pid])
        bar = worker_bars[slot]
        state = slot_state.setdefault(slot, {"key": None, "last_step": 0})
        if event == "done":
            state["key"] = None
            state["last_step"] = 0
            bar.set_description(f"Worker {slot + 1} (idle)")
            return
        if event != "step":
            return
        iteration_i = int(evt.get("iteration", -1))
        particle_i = int(evt.get("particle", -1))
        case_idx = int(evt.get("case_index", -1))
        key = (iteration_i, particle_i, case_idx)
        step = max(int(evt.get("step", 0)), 0)
        total = max(int(evt.get("total", 0)), 0)
        if state.get("key") != key:
            state["key"] = key
            state["last_step"] = 0
            bar.reset(total=max(total, 1))
            case_name = str(evt.get("case_name", "case") or "case")
            bar.set_description(
                f"Worker {slot + 1} (iter {iteration_i + 1}/{int(opt.iterations) + 1}, p{particle_i}, {case_name})"
            )
        if total > 0 and int(bar.total) != total:
            bar.total = total
        last_step = int(state.get("last_step", 0))
        if step > last_step:
            bar.update(step - last_step)
            state["last_step"] = step

    def _write_checkpoint(*, completed_iteration: int, status: str = "running") -> None:
        _write_pso_state(
            checkpoint_path,
            {
                "schema_version": 1,
                "status": status,
                "contract_digest": contract_digest,
                "completed_iteration": int(completed_iteration),
                "rng_state": rng.bit_generator.state,
                "x": x.tolist(),
                "v": v.tolist(),
                "pbest_x": pbest_x.tolist(),
                "pbest_f": pbest_f.tolist(),
                "gbest_x": gbest_x.tolist(),
                "gbest_f": float(gbest_f),
                "history": list(history),
                "evaluations": all_evaluations,
                "parallel_fallback_reason": optimization_execution_meta.get("parallel_fallback_reason"),
            },
        )
        partial_evaluations.clear()
        _write_pso_state(
            partial_path,
            {
                "schema_version": 1,
                "status": "checkpointed",
                "contract_digest": contract_digest,
                "completed_iteration": int(completed_iteration),
                "evaluations": [],
            },
        )

    try:
        if checkpoint_payload is not None:
            try:
                x = np.array(checkpoint_payload["x"], dtype=float).reshape((particles, dim))
                v = np.array(checkpoint_payload["v"], dtype=float).reshape((particles, dim))
                pbest_x = np.array(checkpoint_payload["pbest_x"], dtype=float).reshape((particles, dim))
                pbest_f = np.array(checkpoint_payload["pbest_f"], dtype=float).reshape((particles,))
                gbest_x = np.array(checkpoint_payload["gbest_x"], dtype=float).reshape((dim,))
                gbest_f = float(checkpoint_payload["gbest_f"])
                history = [float(item) for item in list(checkpoint_payload["history"])]
                rng.bit_generator.state = dict(checkpoint_payload["rng_state"])
                completed_iteration = int(checkpoint_payload["completed_iteration"])
                start_iteration = completed_iteration + 1
                fallback_reason = checkpoint_payload.get("parallel_fallback_reason")
                if fallback_reason:
                    optimization_execution_meta["parallel_fallback_reason"] = str(fallback_reason)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid controller-bench PSO checkpoint: {checkpoint_path}: {exc}") from exc
        else:
            evaluations = _evaluate_particle_swarm(
                suite=suite,
                cases=list(suite.cases),
                base_variant=base_variant,
                swarm=x,
                iteration=0,
                workers=workers,
                particle_done_callback=_on_particle_done,
                progress_event_callback=_on_worker_progress if worker_bars else None,
                execution_meta=optimization_execution_meta,
            )
            all_evaluations.extend(evaluations)
            f = np.array([float(item["cost"]) for item in evaluations], dtype=float)
            pbest_x = x.copy()
            pbest_f = f.copy()
            g_idx = int(np.argmin(pbest_f))
            gbest_x = pbest_x[g_idx].copy()
            gbest_f = float(pbest_f[g_idx])
            history = [gbest_f]
            start_iteration = 1
            _write_checkpoint(completed_iteration=0)

        for iteration in range(start_iteration, int(opt.iterations) + 1):
            r1 = rng.random((particles, dim))
            r2 = rng.random((particles, dim))
            v = opt.inertia * v + opt.cognitive * r1 * (pbest_x - x) + opt.social * r2 * (gbest_x[None, :] - x)
            x = np.clip(x + v, lb, ub)
            evaluations = _evaluate_particle_swarm(
                suite=suite,
                cases=list(suite.cases),
                base_variant=base_variant,
                swarm=x,
                iteration=iteration,
                workers=workers,
                particle_done_callback=_on_particle_done,
                progress_event_callback=_on_worker_progress if worker_bars else None,
                execution_meta=optimization_execution_meta,
            )
            all_evaluations.extend(evaluations)
            f = np.array([float(item["cost"]) for item in evaluations], dtype=float)
            improved = f < pbest_f
            if np.any(improved):
                pbest_x[improved, :] = x[improved, :]
                pbest_f[improved] = f[improved]
                g_idx = int(np.argmin(pbest_f))
                gbest_x = pbest_x[g_idx].copy()
                gbest_f = float(pbest_f[g_idx])
            history.append(gbest_f)
            _write_checkpoint(completed_iteration=iteration)
        _write_checkpoint(completed_iteration=int(opt.iterations), status="completed")
    finally:
        for bar in worker_bars:
            try:
                bar.close()
            except Exception:
                pass
        if pso_bar is not None:
            try:
                if pso_bar.n < total_particle_evals:
                    pso_bar.update(total_particle_evals - pso_bar.n)
                pso_bar.close()
            except Exception:
                pass
        restore_env_vars(thread_env_prev)

    best_variant = _optimized_variant_from_vector(
        base_variant,
        opt.parameters,
        gbest_x,
        name=f"{base_variant.name}__pso_best",
        description="Best controller parameters found by controller-bench PSO.",
    )
    best_variant_yaml = yaml.safe_dump(
        {
            "name": best_variant.name,
            "description": best_variant.description,
            "controller": deepcopy(best_variant.pointer),
            "pointers": [
                {
                    "object_id": pointer.object_id,
                    "slot": pointer.slot,
                    "controller": deepcopy(pointer.pointer),
                }
                for pointer in best_variant.pointers
            ],
            "overrides": [{"path": override.path, "value": deepcopy(override.value)} for override in best_variant.overrides],
        },
        sort_keys=False,
    )
    best_runs = [_run_single_bench_case(suite=suite, case=case, variant=best_variant) for case in suite.cases]
    variant_summaries = []
    run_count = len(best_runs)
    passed_runs = sum(1 for run in best_runs if bool(run.get("passed", False)))
    variant_summaries.append(
        {
            "variant_name": best_variant.name,
            "description": best_variant.description,
            "run_count": run_count,
            "passed_runs": passed_runs,
            "pass_rate": (float(passed_runs) / float(run_count)) if run_count else 0.0,
            "metric_means": {},
            "objective_pass_rates": {},
        }
    )
    for metric_name in sorted({name for run in best_runs for name in dict(run.get("metrics", {}) or {}).keys()}):
        values = []
        for run in best_runs:
            value = _safe_metric_float(dict(run.get("metrics", {}) or {}).get(metric_name))
            if value is not None and math.isfinite(value):
                values.append(value)
        if values:
            variant_summaries[0]["metric_means"][metric_name] = float(sum(values) / len(values))
    objective_names = []
    for run in best_runs:
        for objective_result in list(run.get("objective_results", []) or []):
            objective_name = str(dict(objective_result or {}).get("name", "") or "").strip()
            if objective_name and objective_name not in objective_names:
                objective_names.append(objective_name)
    for objective_name in objective_names:
        hits = 0
        passes = 0
        for run in best_runs:
            for objective_result in list(run.get("objective_results", []) or []):
                if str(dict(objective_result or {}).get("name", "") or "").strip() != objective_name:
                    continue
                hits += 1
                if bool(dict(objective_result or {}).get("passed", False)):
                    passes += 1
        if hits:
            variant_summaries[0]["objective_pass_rates"][objective_name] = float(passes) / float(hits)
    result = {
        "suite_name": suite.suite_name,
        "description": suite.description,
        "config_path": str(source_config_path or ""),
        "controller_target": {
            "object_id": suite.controller_target.object_id,
            "slot": suite.controller_target.slot,
        },
        "variants": [_variant_report_entry(best_variant)],
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
        "runs": best_runs,
        "variant_summaries": variant_summaries,
        "execution": {
            "parallel_enabled": bool(
                int(optimization_hierarchy.campaign_workers) > 1
                and not optimization_execution_meta.get("parallel_fallback_reason")
            ),
            "parallel_requested": True,
            "parallel_workers": (
                int(optimization_hierarchy.campaign_workers)
                if not optimization_execution_meta.get("parallel_fallback_reason")
                else 1
            ),
            "parallel_fallback_reason": optimization_execution_meta.get("parallel_fallback_reason"),
            "parallel_backend": suite.parallel_backend if not optimization_execution_meta.get("parallel_fallback_reason") else "serial",
            "hierarchical_execution": optimization_hierarchy.payload(),
        },
        "plot_mode": suite.plot_mode,
        "leaderboards": {
            "relative_rendezvous": _build_relative_rendezvous_leaderboards(best_runs, [best_variant]),
            "rocket_orbit_insertion": _build_rocket_insertion_leaderboards(best_runs, [best_variant]),
        },
        "constraint_summary": _aggregate_constraint_summary(best_runs),
        "optimization": {
            "enabled": True,
            "algorithm": "pso",
            "base_variant": base_variant.name,
            "particles": particles,
            "requested_particles": requested_particles,
            "iterations": int(opt.iterations),
            "workers": workers,
            "worker_cap": worker_cap,
            "best_cost": gbest_f,
            "best_parameters": _optimization_parameter_values(opt.parameters, gbest_x),
            "best_vector": gbest_x.tolist(),
            "best_vector_names": parameter_vector_names,
            "best_variant_yaml": best_variant_yaml,
            "history_best_cost": history,
            "evaluations": all_evaluations,
            "checkpoint": {
                "path": str(checkpoint_path),
                "partial_evaluations_path": str(partial_path),
                "contract_digest": contract_digest,
                "resumed": bool(resumed_from_checkpoint),
                "completed_iteration": int(opt.iterations),
            },
            "cost_mode": "relative_rendezvous_cost" if opt.cost is not None else "generic_objective_score",
            "cost": (
                {
                    "object_id": opt.cost.object_id,
                    "reference_object_id": opt.cost.reference_object_id,
                    "terminal_state_weight": opt.cost.terminal_state_weight,
                    "fuel_weight": opt.cost.fuel_weight,
                    "time_weight": opt.cost.time_weight,
                    "rendezvous_radius_km": opt.cost.rendezvous_radius_km,
                }
                if opt.cost is not None
                else {}
            ),
        },
    }
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"] = {
        "pso_checkpoint_json": str(checkpoint_path),
        "pso_partial_evaluations_json": str(partial_path),
    }
    try:
        from sim.interchange.adapters.controller_lab import emit_controller_optimization_scenario_patches

        patch_emission = emit_controller_optimization_scenario_patches(
            result,
            source_suite=source_config_path,
            output_dir=suite.output_dir / "scenario_patches",
        )
    except (OSError, ValueError) as exc:
        patch_emission = {"status": "not_emitted", "reason": str(exc), "selection_required": True}
    result["optimization"]["scenario_patch_emission"] = patch_emission
    if patch_emission.get("status") == "emitted":
        result["artifacts"]["best_controller_variant_yaml"] = patch_emission["standalone_variant_path"]
        result["artifacts"]["scenario_patch_index_json"] = patch_emission["index_path"]
    plot_artifacts = render_controller_bench_visualizations(result, suite.output_dir, suite.plot_mode)
    if plot_artifacts:
        result["artifacts"].update(plot_artifacts)
    result["artifacts"].update(write_controller_bench_reports(result, suite.output_dir))
    return result

__all__ = [name for name in globals() if not name.startswith("__")]
