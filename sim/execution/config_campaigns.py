# ruff: noqa: F401,F403,F405,I001
from .campaign_common import *
from sim.config.scenario.loader import _parse_yaml_mapping, load_simulation_yaml_bytes
from sim.execution.campaign_retention import (
    configure_iteration,
    extract_iteration,
    preflight_campaign_retention,
    retire_iteration_review,
)
from sim.execution.study import analysis_study_type
from sim.execution.thread_campaign import ThreadCampaignExecutor

@dataclass(frozen=True)
class ConfigCampaignRunSpec:
    """One scenario config scheduled by a config-queue campaign."""

    index: int
    name: str
    config_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigCampaignSpec:
    """Thin Pro campaign that queues existing scenario YAML configs."""

    name: str
    description: str
    config_path: Path
    runs: tuple[ConfigCampaignRunSpec, ...]
    execution: dict[str, Any]
    outputs: dict[str, Any]


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_campaign_input_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_campaign_output_root(
    raw_path: str | Path,
    *,
    campaign_name: str,
    path_policy: ConfigPathPolicy,
) -> Path:
    text = str(raw_path or "").strip()
    return path_policy.resolve_output_dir(text or f"outputs/campaigns/{campaign_name}", purpose="outputs.root")


def load_config_campaign(
    campaign_path: str | Path,
    *,
    _content: bytes | None = None,
) -> ConfigCampaignSpec:
    """Load a Pro config-queue campaign YAML file."""

    source = Path(campaign_path).expanduser().resolve()
    content = source.read_bytes() if _content is None else _content
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Campaign YAML must be UTF-8: {source}") from exc
    raw = _parse_yaml_mapping(content, source)

    name = str(raw.get("name") or source.stem).strip() or source.stem
    description = str(raw.get("description", "") or "").strip()
    execution_raw = raw.get("execution", {}) or {}
    outputs_raw = raw.get("outputs", {}) or {}
    if not isinstance(execution_raw, dict):
        raise ValueError("campaign execution must be a mapping/object.")
    if not isinstance(outputs_raw, dict):
        raise ValueError("campaign outputs must be a mapping/object.")

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("campaign runs must be a non-empty list.")

    runs: list[ConfigCampaignRunSpec] = []
    for idx, item in enumerate(runs_raw):
        if isinstance(item, str):
            entry = {"config": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            raise ValueError(f"campaign runs[{idx}] must be a config path or mapping/object.")
        if not _truthy(entry.get("enabled"), default=True):
            continue
        config_text = str(entry.get("config", "") or "").strip()
        if not config_text:
            raise ValueError(f"campaign runs[{idx}].config is required.")
        config_path = _resolve_campaign_input_path(config_text, base_dir=source.parent)
        run_name = str(entry.get("name") or config_path.stem).strip() or config_path.stem
        metadata = {k: v for k, v in entry.items() if k not in {"config", "name", "enabled"}}
        runs.append(ConfigCampaignRunSpec(index=len(runs), name=run_name, config_path=config_path, metadata=metadata))

    if not runs:
        raise ValueError("campaign has no enabled runs.")

    execution = dict(execution_raw)
    mode = str(execution.get("mode", "serial") or "serial").strip().lower()
    if mode not in {"serial", "parallel"}:
        raise ValueError("campaign execution.mode must be 'serial' or 'parallel'.")
    execution["mode"] = mode
    backend = str(execution.get("backend", "process_pool") or "process_pool").strip().lower()
    if backend not in {"process_pool", "thread_pool"}:
        raise ValueError("campaign execution.backend must be 'process_pool' or 'thread_pool'.")
    execution["backend"] = backend
    execution["max_workers"] = int(max(int(execution.get("max_workers", 0) or 0), 0))
    execution["validate_first"] = _truthy(execution.get("validate_first"), default=True)
    execution["stop_on_failure"] = _truthy(execution.get("stop_on_failure"), default=True)

    outputs = dict(outputs_raw)
    path_policy = ConfigPathPolicy.default(config_path=source)
    outputs["root"] = str(
        _resolve_campaign_output_root(outputs.get("root", ""), campaign_name=name, path_policy=path_policy)
    )
    outputs["include_payloads"] = _truthy(outputs.get("include_payloads"), default=False)
    raw_retention = outputs.get("campaign_retention")
    if raw_retention is not None and not isinstance(raw_retention, dict):
        raise ValueError("campaign outputs.campaign_retention must be a mapping/object.")
    retention = dict(raw_retention or {})
    preflight_campaign_retention(retention, len(runs))
    if "full_on_failure" in retention:
        raise ValueError(
            "campaign outputs.campaign_retention.full_on_failure is unsupported for config queues; "
            "use mode: full or full_iterations to retain child review stores."
        )
    if retention and retention.get("mode", "stats") != "full" and outputs["include_payloads"]:
        raise ValueError("campaign outputs.include_payloads cannot be true with compact campaign_retention.")
    if "campaign_retention" in outputs_raw:
        outputs["campaign_retention"] = retention

    return ConfigCampaignSpec(
        name=name,
        description=description,
        config_path=source,
        runs=tuple(runs),
        execution=execution,
        outputs=outputs,
    )


def _validate_campaign_run(
    spec: ConfigCampaignRunSpec,
    *,
    import_plugins: bool = True,
    _loaded_configs: dict[Path, tuple[str, SimulationScenarioConfig]] | None = None,
    retention_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    cfg = None
    try:
        guarded_config_bytes = spec.config_path.read_bytes()
        digest_before = hashlib.sha256(guarded_config_bytes).hexdigest()
        # ``load_simulation_yaml`` already returns the normalized scenario and
        # retains its source path.  Re-normalizing ``to_dict()`` repeated the
        # full contract traversal and discarded that provenance.
        cfg = load_simulation_yaml_bytes(guarded_config_bytes, source_path=spec.config_path)
        if retention_policy and analysis_study_type(cfg) != "single_run":
            raise ValueError(
                "queue-level campaign_retention supports single-run scenarios; "
                "set outputs.campaign_retention in this study's scenario YAML instead"
            )
        if retention_policy and cfg.outputs.campaign_retention:
            raise ValueError(
                "campaign_retention is set in both the queue and scenario YAML; choose one policy"
            )
        if analysis_study_type(cfg) == "single_run" and cfg.outputs.campaign_retention:
            raise ValueError("single-run scenario retention belongs in the queue campaign outputs")
        digest_after = hashlib.sha256(spec.config_path.read_bytes()).hexdigest()
        if digest_before != digest_after:
            raise ValueError(
                f"campaign scenario changed while it was being validated: {spec.config_path}"
            )
        if _loaded_configs is not None:
            _loaded_configs[spec.config_path] = (
                digest_after,
                cfg,
            )
    except Exception as exc:
        errors.append(str(exc))
    if cfg is not None and import_plugins:
        strict_plugins = bool(cfg.simulator.plugin_validation.get("strict", True))
        plugin_errors = validate_scenario_plugins(cfg, import_plugins=True)
        if plugin_errors and strict_plugins:
            errors.extend(plugin_errors)
    return {
        "index": int(spec.index),
        "name": spec.name,
        "config_path": str(spec.config_path),
        "scenario_name": str(getattr(cfg, "scenario_name", "") or "") if cfg is not None else "",
        "output_dir": str(getattr(getattr(cfg, "outputs", None), "output_dir", "") or "") if cfg is not None else "",
        "valid": not errors,
        "errors": errors,
    }


def validate_config_campaign(
    campaign_path: str | Path,
    *,
    import_plugins: bool = True,
    _campaign: ConfigCampaignSpec | None = None,
    _campaign_sha256: str | None = None,
    _loaded_configs: dict[Path, tuple[str, SimulationScenarioConfig]] | None = None,
) -> dict[str, Any]:
    """Validate a config-queue campaign and its scheduled scenario configs."""

    require_pro_feature(FEATURE_CAMPAIGNS)
    errors: list[str] = []
    try:
        if _campaign is not None and _campaign_sha256 is not None:
            current_sha256 = hashlib.sha256(_campaign.config_path.read_bytes()).hexdigest()
            if current_sha256 != _campaign_sha256:
                raise ValueError("campaign file changed after it was loaded; retry with a stable file")
            campaign = _campaign
        else:
            campaign = load_config_campaign(campaign_path)
    except Exception as exc:
        return {
            "valid": False,
            "campaign_path": str(Path(campaign_path).expanduser()),
            "name": "",
            "description": "",
            "run_count": 0,
            "runs": [],
            "errors": [str(exc)],
        }

    runs = [
        _validate_campaign_run(
            run,
            import_plugins=import_plugins,
            _loaded_configs=_loaded_configs,
            retention_policy=dict(campaign.outputs.get("campaign_retention", {}) or {}),
        )
        for run in campaign.runs
    ]
    for run in runs:
        errors.extend(f"{run['name']}: {err}" for err in list(run.get("errors", []) or []))
    output_owners: dict[Path, str] = {}
    for run in runs:
        if not bool(run.get("valid", False)):
            continue
        output_text = str(run.get("output_dir", "") or "").strip()
        if not output_text:
            continue
        output = ConfigPathPolicy.default(config_path=run["config_path"]).resolve_output_dir(
            output_text,
            purpose=f"campaign run {run['name']} output_dir",
        )
        if campaign.outputs.get("campaign_retention") and output == Path(campaign.outputs["root"]):
            errors.append(f"{run['name']}: output_dir must differ from campaign outputs.root when retention is set")
        prior = output_owners.get(output)
        if prior is not None:
            errors.append(
                f"{run['name']}: output_dir collides with campaign run {prior!r}: {output}"
            )
        else:
            output_owners[output] = str(run["name"])
    return {
        "valid": not errors,
        "campaign_path": str(campaign.config_path),
        "name": campaign.name,
        "description": campaign.description,
        "run_count": len(campaign.runs),
        "execution": dict(campaign.execution),
        "outputs": dict(campaign.outputs),
        "runs": runs,
        "errors": errors,
    }


def _output_index_from_payload(payload: dict[str, Any]) -> str:
    run = dict(payload.get("run", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    for candidate in (
        run.get("output_index_md"),
        payload.get("output_index_md"),
        artifacts.get("output_index_md"),
        artifacts.get("index_md"),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    output_dir = str(run.get("output_dir") or payload.get("output_dir") or "").strip()
    if output_dir:
        candidate_path = Path(output_dir) / "index.md"
        if candidate_path.exists():
            return str(candidate_path)
    return ""


def _output_dir_from_payload(payload: dict[str, Any]) -> str:
    run = dict(payload.get("run", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    for candidate in (run.get("output_dir"), payload.get("output_dir"), artifacts.get("output_dir")):
        text = str(candidate or "").strip()
        if text:
            return text
    index_path = _output_index_from_payload(payload)
    return str(Path(index_path).parent) if index_path else ""


def _short_campaign_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if str(dict(payload.get("analysis", {}) or {}).get("study_type", "")) == "sensitivity":
        analysis = dict(payload.get("analysis", {}) or {})
        return {
            "workflow": "sensitivity",
            "scenario_name": str(payload.get("scenario_name", "")),
            "run_count": int(analysis.get("run_count", len(payload.get("runs", []) or [])) or 0),
            "output_dir": _output_dir_from_payload(payload),
            "output_index_md": _output_index_from_payload(payload),
        }
    if bool(dict(payload.get("monte_carlo", {}) or {}).get("enabled", False)):
        mc = dict(payload.get("monte_carlo", {}) or {})
        agg = dict(payload.get("aggregate_stats", {}) or {})
        return {
            "workflow": "monte_carlo",
            "scenario_name": str(payload.get("scenario_name", "")),
            "run_count": int(mc.get("iterations", len(payload.get("runs", []) or [])) or 0),
            "pass_rate": agg.get("pass_rate"),
            "output_dir": _output_dir_from_payload(payload),
            "output_index_md": _output_index_from_payload(payload),
        }

    run = dict(payload.get("run", {}) or {})
    thrust = dict(run.get("thrust_stats", {}) or {})
    total_dv_m_s = 0.0
    for stats in thrust.values():
        try:
            total_dv_m_s += float(dict(stats or {}).get("total_dv_m_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return {
        "workflow": "single_run",
        "scenario_name": str(payload.get("scenario_name", run.get("scenario_name", "")) or ""),
        "duration_s": run.get("duration_s"),
        "samples": run.get("samples"),
        "terminated_early": bool(run.get("terminated_early", False)),
        "termination_reason": run.get("termination_reason"),
        "total_dv_m_s": total_dv_m_s,
        "output_dir": _output_dir_from_payload(payload),
        "output_index_md": _output_index_from_payload(payload),
    }


def _run_config_campaign_worker(task: dict[str, Any]) -> dict[str, Any]:
    from sim.execution.service import SimulationExecutionService

    started = time.perf_counter()
    spec = dict(task.get("spec", {}) or {})
    run_index = int(spec.get("index", 0))
    run_name = str(spec.get("name", ""))
    config_path = str(spec.get("config_path", ""))
    expected_config_sha256 = str(spec.get("expected_config_sha256", "") or "")
    metadata = dict(spec.get("metadata", {}) or {})
    retention_policy = dict(task.get("retention_policy", {}) or {})
    progress_queue = task.get("progress_queue")
    if progress_queue is None:
        progress_queue = worker_progress_queue()
    emit_every = int(task.get("progress_emit_every", 20) or 20)
    emit_every = max(1, emit_every)
    last_emit = -(10**9)
    cancel_event = task.get("cancel_event")

    def _emit_progress(event: str, **payload: Any) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put(
                {
                    "event": event,
                    "pid": int(os.getpid()),
                    "iteration": run_index,
                    "run_index": run_index,
                    "run_name": run_name,
                    **payload,
                }
            )
        except Exception:
            pass

    def _on_step(step: int, total: int) -> None:
        nonlocal last_emit
        if cancel_event is not None and cancel_event.is_set():
            from sim.execution.thread_campaign import ThreadCampaignCancelled

            raise ThreadCampaignCancelled("Thread campaign worker cancelled.")
        s = max(int(step), 0)
        t = max(int(total), 0)
        should_emit = (s == 0) or (t > 0 and s >= t) or (s - last_emit >= emit_every)
        if not should_emit:
            return
        last_emit = s
        _emit_progress("step", step=int(s), total=int(t))

    try:
        if not expected_config_sha256:
            raise ValueError("campaign task is missing its validated config digest")
        guarded_config_bytes = Path(config_path).read_bytes()
        actual_config_sha256 = hashlib.sha256(guarded_config_bytes).hexdigest()
        if actual_config_sha256 != expected_config_sha256:
            raise ValueError(
                f"campaign scenario changed after task preparation: {config_path}"
            )
        _emit_progress("start", step=0, total=0)
        service = SimulationExecutionService()
        cfg = service.load_config_bytes(guarded_config_bytes, source_path=config_path)
        if retention_policy:
            if analysis_study_type(cfg) != "single_run":
                raise ValueError("queue-level campaign_retention supports single-run scenarios only")
            from sim.config import scenario_config_from_dict

            child_config = cfg.to_dict()
            configure_iteration(child_config, policy=retention_policy)
            cfg = scenario_config_from_dict(child_config)
        payload = service.run_config(
            cfg,
            source_path=config_path,
            step_callback=_on_step if progress_queue is not None else None,
        )
        _emit_progress("done")
        result = {
            "index": run_index,
            "name": run_name,
            "config_path": config_path,
            "metadata": metadata,
            "status": "passed",
            "wall_time_s": float(time.perf_counter() - started),
            "summary": _short_campaign_payload_summary(dict(payload or {})),
        }
        if retention_policy:
            output_dir = Path(cfg.outputs.output_dir).expanduser().resolve()
            result["retained_evidence"] = extract_iteration(output_dir, retention_policy)
            review_path = output_dir / "review" / "run.sqlite"
            keep_full = retention_policy.get("mode", "stats") == "full" or run_index in retention_policy.get(
                "full_iterations", []
            )
            if not keep_full:
                retire_iteration_review(
                    output_dir,
                    expected_sha256=result["retained_evidence"]["source_review_sha256"],
                )
            result["full_review"] = str(review_path) if keep_full else None
        if bool(task.get("include_payload", False)):
            result["payload"] = payload
        return result
    except Exception as exc:
        _raise_if_cancellation(exc)
        _emit_progress("done")
        return {
            "index": run_index,
            "name": run_name,
            "config_path": config_path,
            "metadata": metadata,
            "status": "failed",
            "wall_time_s": float(time.perf_counter() - started),
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {},
        }


def _campaign_run_task(
    spec: ConfigCampaignRunSpec,
    *,
    include_payload: bool,
    expected_config_sha256: str,
    retention_policy: dict[str, Any] | None = None,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "spec": {
            "index": int(spec.index),
            "name": spec.name,
            "config_path": str(spec.config_path),
            "metadata": dict(spec.metadata),
            "expected_config_sha256": str(expected_config_sha256),
        },
        "include_payload": bool(include_payload),
        "retention_policy": dict(retention_policy or {}),
        "config_dict": None if config_dict is None else dict(config_dict),
    }


def _write_config_campaign_artifacts(campaign: ConfigCampaignSpec, payload: dict[str, Any]) -> dict[str, str]:
    outdir = Path(str(campaign.outputs.get("root", ""))).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    summary_json = outdir / "campaign_summary.json"
    runs_csv = outdir / "campaign_runs.csv"
    summary_md = outdir / "campaign_summary.md"

    rows = []
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        rows.append(
            {
                "index": int(run.get("index", 0)),
                "name": str(run.get("name", "")),
                "status": str(run.get("status", "")),
                "workflow": str(summary.get("workflow", "")),
                "scenario_name": str(summary.get("scenario_name", "")),
                "config_path": str(run.get("config_path", "")),
                "output_dir": str(summary.get("output_dir", "")),
                "total_dv_m_s": summary.get("total_dv_m_s", ""),
                "duration_s": summary.get("duration_s", ""),
                "wall_time_s": run.get("wall_time_s", ""),
                "error": str(run.get("error", "")),
                "metadata_json": json.dumps(dict(run.get("metadata", {}) or {}), sort_keys=True),
            }
        )
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "name",
                "status",
                "workflow",
                "scenario_name",
                "config_path",
                "output_dir",
                "total_dv_m_s",
                "duration_s",
                "wall_time_s",
                "error",
                "metadata_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {campaign.name} Campaign",
        "",
        campaign.description,
        "",
        f"- Runs: {payload['summary']['run_count']}",
        f"- Passed: {payload['summary']['passed_count']}",
        f"- Failed: {payload['summary']['failed_count']}",
        f"- Execution: {payload['execution']['mode']}",
        "",
        "| Run | Status | Workflow | Scenario | Output |",
        "|---|---:|---|---|---|",
    ]
    for run in list(payload.get("runs", []) or []):
        summary = dict(run.get("summary", {}) or {})
        output = str(summary.get("output_index_md") or summary.get("output_dir") or "")
        lines.append(
            f"| {run.get('name', '')} | {run.get('status', '')} | "
            f"{summary.get('workflow', '')} | {summary.get('scenario_name', '')} | {output} |"
        )
    summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "output_dir": str(outdir),
        "summary_json": str(summary_json),
        "runs_csv": str(runs_csv),
        "summary_md": str(summary_md),
        "output_index_md": str(summary_md),
    }
