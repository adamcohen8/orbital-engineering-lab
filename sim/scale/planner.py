from __future__ import annotations

import json
import math
import platform
import shutil
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import yaml

from sim.numeric_backend import normalize_numeric_backend

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows hosts.
    resource = None  # type: ignore[assignment]

from sim.resource_limits import current_resource_snapshot
from sim.scale._output import bind_output_writer, staged_sqlite_store
from sim.scale.onp_identity import onp_materialization_signature
from sim.scale.store import (
    StoreSummary,
    count_current_conjunction_candidates,
    count_matching_current_propagation_products,
    count_screening_windows_for_current_products,
    estimate_matching_pair_window_count,
    initialize_store,
    load_current_tle_states,
    pending_candidate_refinement_count,
    pending_hpop_handoff_count,
    pending_onp_refinement_materialization_count,
    summarize_onp_refinement_execution_queue,
    summarize_store,
)
from sim.scale.windows import build_propagation_windows
from sim.security import ConfigPathPolicy
from sim.security.output_writes import OutputWriter


@dataclass(frozen=True)
class ScalePlannerConfig:
    name: str
    description: str
    config_path: Path
    mode: str
    store_path: Path
    store_backend: str
    output_dir: Path
    current_state_policy: str
    object_ids: tuple[str, ...]
    tags: tuple[str, ...]
    propagation_model: str
    propagation_backend: str
    output_frame: str
    start_jd_utc: float
    duration_s: float
    step_s: float
    chunk_duration_s: float
    screening_mode: str
    include_previous_close_pairs: bool
    close_pair_lookback_run_id: str
    distance_threshold_km: float
    promotion_limit: int
    refinement_step_s: float
    refinement_half_window_s: float
    refinement_limit: int
    ranking_policy: str
    hpop_handoff_enabled: bool
    hpop_handoff_output_dir: Path
    hpop_handoff_limit: int
    hpop_handoff_half_window_s: float
    force_model_profile: str
    sensitivity_enabled: bool
    sensitivity_campaign_type: str
    sensitivity_target: str
    sensitivity_uncertainty_interpretation: str
    sensitivity_seed: int
    sensitivity_trial_count: int
    sensitivity_sweep_mode: str
    sensitivity_parameters: tuple[dict[str, Any], ...]
    sensitivity_derived_parameters: tuple[dict[str, Any], ...]
    sensitivity_frame: dict[str, Any]
    sensitivity_trial_builder: dict[str, Any]
    sensitivity_outputs: dict[str, Any]
    sensitivity_output_dir: Path
    monte_carlo_enabled: bool
    monte_carlo_campaign_type: str
    monte_carlo_target: str
    monte_carlo_uncertainty_interpretation: str
    monte_carlo_seed: int
    monte_carlo_trial_count: int
    monte_carlo_parameters: tuple[dict[str, Any], ...]
    monte_carlo_derived_parameters: tuple[dict[str, Any], ...]
    monte_carlo_frame: dict[str, Any]
    monte_carlo_trial_builder: dict[str, Any]
    monte_carlo_outputs: dict[str, Any]
    monte_carlo_output_dir: Path
    planner_limits: dict[str, Any]
    screening_distance_comparison: str = "exact_sqrt"
    screening_numeric_backend: str = "rust"
    propagation_workers: int = 0

    @property
    def onp_handoff_enabled(self) -> bool:
        return self.hpop_handoff_enabled

    @property
    def onp_handoff_output_dir(self) -> Path:
        return self.hpop_handoff_output_dir

    @property
    def onp_handoff_limit(self) -> int:
        return self.hpop_handoff_limit

    @property
    def onp_handoff_half_window_s(self) -> float:
        return self.hpop_handoff_half_window_s


class ScalePlannerLimitError(RuntimeError):
    """Raised when a configured Scale execution ceiling is exceeded."""


_ACTIVE_SCALE_LIMITS: ContextVar[tuple[ScalePlannerConfig, str, float] | None] = ContextVar(
    "oel_active_scale_limits",
    default=None,
)


def check_active_scale_planner_limits() -> None:
    """Check the owner-side planner ceilings for the active operation.

    Parallel owners must call this while futures are pending, rather than only
    after a result arrives.  The check is cooperative: it cannot impose an OS
    memory quota or forcibly stop a Python thread/native kernel, so owners must
    cancel queued work and let active thread workers reach their cancellation
    checks before shutdown.
    """
    active = _ACTIVE_SCALE_LIMITS.get()
    if active is None:
        return
    cfg, operation, started_s = active
    _check_scale_planner_limits(cfg, operation=operation, started_s=started_s)


def enforce_scale_planner_limits(operation: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a Scale config-path operation in its configured hard resource ceilings."""

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _wrapped(config_path: str | Path, *args: Any, **kwargs: Any) -> Any:
            cfg = load_scale_planner_config(config_path)
            started_s = time.monotonic()
            _check_scale_planner_limits(cfg, operation=operation, started_s=started_s)
            token = _ACTIVE_SCALE_LIMITS.set((cfg, operation, started_s))
            try:
                result = func(config_path, *args, **kwargs)
                _check_scale_planner_limits(cfg, operation=operation, started_s=started_s)
                return result
            finally:
                _ACTIVE_SCALE_LIMITS.reset(token)

        return _wrapped

    return _decorate


def _check_scale_planner_limits(
    cfg: ScalePlannerConfig,
    *,
    operation: str,
    started_s: float,
) -> None:
    runtime_limit = _planner_limit(cfg, "max_runtime_s")
    memory_limit = _planner_limit(cfg, "max_memory_mb")
    storage_limit = _planner_limit(cfg, "max_storage_mb")
    elapsed_s = max(time.monotonic() - float(started_s), 0.0)
    if runtime_limit is not None and elapsed_s > runtime_limit:
        raise ScalePlannerLimitError(
            f"Scale {operation} exceeded planner.max_runtime_s: {elapsed_s:.3f} s > {runtime_limit:.3f} s."
        )
    if memory_limit is not None:
        rss_mb = _current_process_rss_mb()
        if rss_mb > memory_limit:
            raise ScalePlannerLimitError(
                f"Scale {operation} exceeds planner.max_memory_mb: {rss_mb:.3f} MB > {memory_limit:.3f} MB."
            )
    if storage_limit is not None:
        storage_mb = _scale_owned_storage_mb(cfg)
        if storage_mb > storage_limit:
            raise ScalePlannerLimitError(
                f"Scale {operation} exceeds planner.max_storage_mb: {storage_mb:.3f} MB > {storage_limit:.3f} MB."
            )


def _planner_limit(cfg: ScalePlannerConfig, name: str) -> float | None:
    value = cfg.planner_limits.get(name)
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"scale_catalog.planner.{name} must be a finite positive number or null.")
    return out


def _current_process_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        process = psutil.Process()
        rss_bytes = float(process.memory_info().rss)
        for child in process.children(recursive=True):
            try:
                rss_bytes += float(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss_bytes / (1024.0 * 1024.0)
    except (ImportError, OSError, AttributeError) as exc:
        if resource is None:
            raise RuntimeError(
                "Scale memory limits require psutil on platforms without the resource module."
            ) from exc
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss += float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        if platform.system() != "Darwin":
            rss *= 1024.0
        return rss / (1024.0 * 1024.0)


def _scale_owned_storage_mb(cfg: ScalePlannerConfig) -> float:
    roots = {
        cfg.store_path,
        cfg.output_dir,
        cfg.hpop_handoff_output_dir,
        cfg.sensitivity_output_dir,
        cfg.monte_carlo_output_dir,
    }
    files: set[Path] = set()
    total = 0
    for root in roots:
        path = Path(root).expanduser().resolve()
        candidates = [path] if path.is_file() else (path.rglob("*") if path.is_dir() else [])
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if not candidate.is_file() or resolved in files:
                    continue
                files.add(resolved)
                total += int(candidate.stat().st_size)
            except OSError:
                continue
    return float(total) / (1024.0 * 1024.0)


def run_dry_run(
    config_path: str | Path,
    *,
    output_dir: Path | None = None,
    writer: OutputWriter | None = None,
) -> dict[str, Any]:
    cfg = load_scale_planner_config(config_path, output_dir=output_dir)
    if cfg.mode != "dry_run":
        raise ValueError("Phase 0 skeleton only supports scale_catalog.mode: dry_run.")

    with staged_sqlite_store(cfg.store_path, writer=writer) as store_path:
        initialize_store(store_path, description=f"OEL Scale store for {cfg.name}")
        store_summary = summarize_store(store_path)
        # All planning reads must use the private staged copy while a
        # descriptor-backed writer is active.  The logical path is restored in
        # the payload below so callers never receive a staging filename.
        payload_cfg = replace(cfg, store_path=store_path)
        payload = build_dry_run_payload(payload_cfg, store_summary=store_summary)
    payload["store"]["path"] = str(cfg.store_path)
    json_path = cfg.output_dir / "scale_planner_dry_run.json"
    md_path = cfg.output_dir / "scale_planner_dry_run.md"
    payload["artifacts"] = {
        "planner_json": str(json_path),
        "planner_markdown": str(md_path),
    }
    serialized_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    serialized_markdown = _render_markdown(payload)
    binding = bind_output_writer(writer, cfg.output_dir)
    if binding is None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(serialized_json, encoding="utf-8")
        md_path.write_text(serialized_markdown, encoding="utf-8")
    else:
        binding.write_text(json_path, serialized_json)
        binding.write_text(md_path, serialized_markdown)
    return payload


def load_scale_planner_config(config_path: str | Path, *, output_dir: Path | None = None) -> ScalePlannerConfig:
    source = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Scale config root must be a mapping: {source}")
    section = raw.get("scale_catalog")
    if not isinstance(section, dict):
        raise ValueError("Scale config requires a scale_catalog mapping.")

    name = str(raw.get("name") or source.stem).strip() or source.stem
    description = str(raw.get("description", "") or "").strip()
    mode = str(section.get("mode", "dry_run") or "dry_run").strip().lower()
    if mode not in {"dry_run", "run"}:
        raise ValueError("scale_catalog.mode must be dry_run or run.")

    path_policy = ConfigPathPolicy.default(config_path=source)
    store_cfg = _dict_section(section, "store")
    store_backend = str(store_cfg.get("backend", "sqlite") or "sqlite").strip().lower()
    if store_backend != "sqlite":
        raise ValueError("Phase 0 skeleton supports only scale_catalog.store.backend: sqlite.")
    store_path = _resolve_output_file(
        store_cfg.get("path", f"outputs/scale/{name}/catalog_store.sqlite"),
        path_policy=path_policy,
        purpose="scale_catalog.store.path",
    )

    planner_cfg = _dict_section(section, "planner")
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else path_policy.resolve_output_dir(
            planner_cfg.get("output_dir", f"outputs/scale/{name}"),
            purpose="scale_catalog.planner.output_dir",
        )
    )

    selection_cfg = _dict_section(section, "selection")
    object_filter = _dict_section(selection_cfg, "object_filter")
    propagation_cfg = _dict_section(section, "propagation")
    screening_cfg = _dict_section(section, "screening")
    refinement_cfg = _dict_section(section, "refinement")
    if "onp_handoff" in section and "hpop_handoff" in section:
        raise ValueError("Use scale_catalog.onp_handoff or deprecated hpop_handoff, not both.")
    hpop_cfg = _dict_section(section, "onp_handoff" if "onp_handoff" in section else "hpop_handoff")
    sensitivity_cfg = _dict_section(section, "sensitivity")
    monte_carlo_cfg = _dict_section(section, "monte_carlo")
    hpop_output_dir = path_policy.resolve_output_dir(
        hpop_cfg.get("output_dir", f"outputs/scale/{name}/handoff"),
        purpose="scale_catalog.onp_handoff.output_dir",
    )
    sensitivity_output_dir = path_policy.resolve_output_dir(
        sensitivity_cfg.get("output_dir", f"outputs/scale/{name}/sensitivity"),
        purpose="scale_catalog.sensitivity.output_dir",
    )
    monte_carlo_output_dir = path_policy.resolve_output_dir(
        monte_carlo_cfg.get("output_dir", f"outputs/scale/{name}/monte_carlo"),
        purpose="scale_catalog.monte_carlo.output_dir",
    )

    propagation_model_raw = str(propagation_cfg.get("model", "ogp") or "ogp").strip().lower()
    if propagation_model_raw not in {"ogp", "sgp4"}:
        raise ValueError("Scale catalog propagation supports scale_catalog.propagation.model: ogp.")
    propagation_model = "ogp"
    propagation_backend = str(propagation_cfg.get("backend", "scalar") or "scalar").strip().lower()
    if propagation_backend not in {"scalar", "numpy", "rust"}:
        raise ValueError("Scale catalog supports propagation.backend scalar, numpy, or rust.")
    propagation_workers = propagation_cfg.get("workers", 0)
    if isinstance(propagation_workers, bool) or not isinstance(propagation_workers, int) or propagation_workers < 0:
        raise ValueError("scale_catalog.propagation.workers must be a nonnegative integer.")
    if propagation_workers and propagation_backend != "rust":
        raise ValueError("scale_catalog.propagation.workers applies only to the rust backend.")

    output_frame = str(propagation_cfg.get("output_frame", "teme") or "teme").strip().lower()
    if output_frame not in {"eci", "teme"}:
        raise ValueError("scale_catalog.propagation.output_frame must be eci or teme.")
    start_jd_utc = _finite_float(
        propagation_cfg.get("start_jd_utc", 0.0) or 0.0,
        name="start_jd_utc",
    )
    step_s = _positive_float(propagation_cfg.get("step_s", 300.0), name="step_s")
    promotion_limit = int(max(int(screening_cfg.get("promotion_limit", 50) or 50), 0))
    sensitivity_enabled = _truthy(sensitivity_cfg.get("enabled"), default=False)
    sensitivity_uncertainty = str(
        sensitivity_cfg.get("uncertainty_interpretation", "analyst_defined_sensitivity") or ""
    ).strip()
    if sensitivity_uncertainty == "covariance_derived":
        raise ValueError(
            "scale_catalog.sensitivity.uncertainty_interpretation covariance_derived is not supported yet."
        )
    monte_carlo_enabled = _truthy(monte_carlo_cfg.get("enabled"), default=False)
    monte_carlo_uncertainty = str(
        monte_carlo_cfg.get("uncertainty_interpretation", "analyst_defined_monte_carlo") or ""
    ).strip()
    if monte_carlo_uncertainty == "covariance_derived":
        raise ValueError(
            "scale_catalog.monte_carlo.uncertainty_interpretation covariance_derived is not supported yet."
        )

    current_state_policy = str(selection_cfg.get("current_state_policy", "latest_ogp_od_or_tle") or "").strip()
    if current_state_policy not in {"latest_ogp_od_or_tle", "latest_sgp4_od_or_tle"}:
        raise ValueError("Scale execution supports selection.current_state_policy: latest_ogp_od_or_tle (legacy latest_sgp4_od_or_tle is accepted).")

    distance_comparison = str(
        screening_cfg.get("distance_comparison", "exact_sqrt") or "exact_sqrt"
    ).strip().lower()
    if distance_comparison not in {"exact_sqrt", "squared_distance"}:
        raise ValueError(
            "scale_catalog.screening.distance_comparison must be exact_sqrt or squared_distance."
        )
    screening_numeric_backend = normalize_numeric_backend(
        screening_cfg.get("numeric_backend", "rust"), field_name="scale_catalog.screening.numeric_backend"
    )
    if screening_numeric_backend not in {"python", "rust"}:
        raise ValueError("scale_catalog.screening.numeric_backend must be python or rust.")

    return ScalePlannerConfig(
        name=name,
        description=description,
        config_path=source,
        mode=mode,
        store_path=store_path,
        store_backend=store_backend,
        output_dir=resolved_output_dir,
        current_state_policy=current_state_policy,
        object_ids=_string_tuple(object_filter.get("object_ids", ())),
        tags=_string_tuple(object_filter.get("tags", ())),
        propagation_model=propagation_model,
        propagation_backend=propagation_backend,
        output_frame=output_frame,
        start_jd_utc=start_jd_utc,
        duration_s=_positive_float(propagation_cfg.get("duration_s", 86400.0), name="duration_s"),
        step_s=step_s,
        chunk_duration_s=_positive_float(propagation_cfg.get("chunk_duration_s", 3600.0), name="chunk_duration_s"),
        screening_mode=str(screening_cfg.get("mode", "sampled_distance") or "sampled_distance").strip().lower(),
        include_previous_close_pairs=_truthy(screening_cfg.get("include_previous_close_pairs"), default=True),
        close_pair_lookback_run_id=str(screening_cfg.get("close_pair_lookback_run_id", "latest") or "latest"),
        distance_threshold_km=_positive_float(
            screening_cfg.get("distance_threshold_km", 25.0), name="distance_threshold_km"
        ),
        promotion_limit=promotion_limit,
        refinement_step_s=_positive_float(refinement_cfg.get("step_s", max(1.0, step_s / 10.0)), name="step_s"),
        refinement_half_window_s=_positive_float(
            refinement_cfg.get("half_window_s", max(step_s, 1.0)), name="half_window_s"
        ),
        refinement_limit=int(max(int(refinement_cfg.get("limit", promotion_limit) or promotion_limit), 0)),
        ranking_policy=str(refinement_cfg.get("ranking_policy", "miss_distance_then_relative_speed") or "").strip(),
        hpop_handoff_enabled=_truthy(hpop_cfg.get("enabled"), default=True),
        hpop_handoff_output_dir=hpop_output_dir,
        hpop_handoff_limit=int(max(int(hpop_cfg.get("limit", promotion_limit) or promotion_limit), 0)),
        hpop_handoff_half_window_s=_positive_float(
            hpop_cfg.get("case_half_window_s", max(600.0, step_s)), name="case_half_window_s"
        ),
        force_model_profile=str(hpop_cfg.get("force_model_profile", "quick_numerical") or "quick_numerical"),
        sensitivity_enabled=sensitivity_enabled,
        sensitivity_campaign_type=str(
            sensitivity_cfg.get("campaign_type", "parameter_sweep") or "parameter_sweep"
        ).strip(),
        sensitivity_target=str(sensitivity_cfg.get("target", "selected_objects") or "selected_objects").strip(),
        sensitivity_uncertainty_interpretation=sensitivity_uncertainty,
        sensitivity_seed=int(sensitivity_cfg.get("seed", 0) or 0),
        sensitivity_trial_count=int(max(int(sensitivity_cfg.get("trial_count", 0) or 0), 0)),
        sensitivity_sweep_mode=str(sensitivity_cfg.get("sweep_mode", "zip") or "zip").strip().lower(),
        sensitivity_parameters=tuple(dict(item) for item in sensitivity_cfg.get("parameters", []) or []),
        sensitivity_derived_parameters=tuple(
            dict(item) for item in sensitivity_cfg.get("derived_parameters", []) or []
        ),
        sensitivity_frame=dict(sensitivity_cfg.get("frame", {}) or {}),
        sensitivity_trial_builder=dict(sensitivity_cfg.get("trial_builder", {}) or {}),
        sensitivity_outputs=dict(sensitivity_cfg.get("outputs", {}) or {}),
        sensitivity_output_dir=sensitivity_output_dir,
        monte_carlo_enabled=monte_carlo_enabled,
        monte_carlo_campaign_type=str(
            monte_carlo_cfg.get("campaign_type", "deployment_dispersion") or "deployment_dispersion"
        ).strip(),
        monte_carlo_target=str(monte_carlo_cfg.get("target", "selected_objects") or "selected_objects").strip(),
        monte_carlo_uncertainty_interpretation=monte_carlo_uncertainty,
        monte_carlo_seed=int(monte_carlo_cfg.get("seed", 0) or 0),
        monte_carlo_trial_count=int(max(int(monte_carlo_cfg.get("trial_count", 0) or 0), 0)),
        monte_carlo_parameters=tuple(dict(item) for item in monte_carlo_cfg.get("parameters", []) or []),
        monte_carlo_derived_parameters=tuple(
            dict(item) for item in monte_carlo_cfg.get("derived_parameters", []) or []
        ),
        monte_carlo_frame=dict(monte_carlo_cfg.get("frame", {}) or {}),
        monte_carlo_trial_builder=dict(monte_carlo_cfg.get("trial_builder", {}) or {}),
        monte_carlo_outputs=dict(monte_carlo_cfg.get("outputs", {}) or {}),
        monte_carlo_output_dir=monte_carlo_output_dir,
        planner_limits=_validated_planner_limits(planner_cfg),
        screening_distance_comparison=distance_comparison,
        screening_numeric_backend=screening_numeric_backend,
        propagation_workers=propagation_workers,
    )


def build_dry_run_payload(cfg: ScalePlannerConfig, *, store_summary: StoreSummary) -> dict[str, Any]:
    snapshot = current_resource_snapshot()
    disk = shutil.disk_usage(cfg.output_dir.parent if cfg.output_dir.parent.exists() else Path.cwd())
    propagation_windows = build_propagation_windows(
        start_jd_utc=cfg.start_jd_utc,
        duration_s=cfg.duration_s,
        step_s=cfg.step_s,
        chunk_duration_s=cfg.chunk_duration_s,
    )
    sample_count_per_chunk = [window.sample_count for window in propagation_windows]
    sample_count_per_object = int(sum(sample_count_per_chunk))
    chunk_count = len(propagation_windows)
    selected_object_ids = resolve_selected_object_ids(cfg)
    selected_object_count = len(selected_object_ids)
    estimated_samples = int(selected_object_count * sample_count_per_object)
    expected_propagation_chunks = int(selected_object_count * chunk_count)
    expected_windows = _expected_propagation_windows(cfg)
    existing_propagation_products = count_matching_current_propagation_products(
        cfg.store_path,
        object_ids=selected_object_ids,
        windows=expected_windows,
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    pending_propagation_chunks = max(expected_propagation_chunks - existing_propagation_products, 0)
    total_conjunction_candidates = int(store_summary.table_counts.get("conjunction_candidates", 0))
    existing_conjunction_candidates = count_current_conjunction_candidates(
        cfg.store_path, object_ids=selected_object_ids
    )
    existing_candidate_refinements = int(store_summary.table_counts.get("candidate_refinements", 0))
    existing_hpop_handoff_packets = int(store_summary.table_counts.get("hpop_handoff_packets", 0))
    existing_onp_refinement_cases = int(store_summary.table_counts.get("onp_refinement_cases", 0))
    estimated_pair_windows = estimate_matching_pair_window_count(
        cfg.store_path,
        object_ids=selected_object_ids,
        windows=expected_windows,
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    existing_screening_windows = count_screening_windows_for_current_products(
        cfg.store_path,
        object_ids=selected_object_ids,
        windows=expected_windows,
        model=cfg.propagation_model,
        backend=cfg.propagation_backend,
        output_frame=cfg.output_frame,
    )
    pending_pair_windows = max(estimated_pair_windows - existing_screening_windows, 0)
    pending_candidate_refinements = pending_candidate_refinement_count(cfg.store_path, object_ids=selected_object_ids)
    pending_hpop_handoff_packets = pending_hpop_handoff_count(cfg.store_path, object_ids=selected_object_ids)
    pending_onp_materializations = pending_onp_refinement_materialization_count(
        cfg.store_path,
        object_ids=selected_object_ids,
        materialization_signature=onp_materialization_signature(
            refinement_step_s=cfg.refinement_step_s,
        ),
    )
    onp_execution = summarize_onp_refinement_execution_queue(cfg.store_path, object_ids=selected_object_ids)
    expected_sensitivity_trials = _estimate_sensitivity_trial_count(cfg)
    expected_monte_carlo_trials = cfg.monte_carlo_trial_count if cfg.monte_carlo_enabled else 0
    previous_close_pair_count = int(
        store_summary.conjunction_candidate_count if cfg.include_previous_close_pairs else 0
    )
    warnings = _warnings(cfg, selected_object_count=selected_object_count)
    planned_jobs = [
        {
            "job_type": "select_current_states",
            "count": 1,
            "status": "planned",
        },
        {
            "job_type": "propagate_ogp_chunk",
            "aliases": ["propagate_sgp4_chunk"],
            "count": pending_propagation_chunks,
            "status": "planned",
        },
        {
            "job_type": "screen_pair_window",
            "count": pending_pair_windows,
            "status": "planned",
            "note": "Phase 1C screens completed OGP product pairs that share a sampled window.",
        },
        {
            "job_type": "refine_candidate_tca",
            "count": pending_candidate_refinements,
            "status": "planned",
            "note": "Phase 1D refines promoted OGP candidates on a tighter sampled grid.",
        },
        {
            "job_type": "rank_candidates",
            "count": 1 if existing_conjunction_candidates else 0,
            "status": "planned",
        },
        {
            "job_type": "write_onp_handoff",
            "aliases": ["write_hpop_handoff"],
            "count": pending_hpop_handoff_packets if cfg.hpop_handoff_enabled else 0,
            "status": "planned",
        },
        {
            "job_type": "write_hpop_handoff",
            "compatibility_alias_for": "write_onp_handoff",
            "deprecated": True,
            "count": pending_hpop_handoff_packets if cfg.hpop_handoff_enabled else 0,
            "status": "planned",
        },
        {
            "job_type": "materialize_onp_refinement",
            "count": pending_onp_materializations if cfg.hpop_handoff_enabled else 0,
            "status": "planned",
            "note": "Generate and validate ordinary ONP scenarios; do not execute them.",
        },
        {
            "job_type": "execute_onp_refinement",
            "count": onp_execution["pending_execution_count"],
            "status": "planned",
            "note": "Resume validated ONP cases and write OGP-vs-ONP review-store comparison evidence.",
        },
        {
            "job_type": "materialize_sensitivity_trials",
            "count": expected_sensitivity_trials if cfg.sensitivity_enabled else 0,
            "status": "planned",
            "note": "Phase 1F materializes analyst-defined sensitivity trials; it does not run trial builders yet.",
        },
        {
            "job_type": "materialize_monte_carlo_trials",
            "count": expected_monte_carlo_trials,
            "status": "planned",
            "note": "Materializes analyst-defined Monte Carlo trials; it does not run trial builders yet.",
        },
    ]
    return {
        "schema_version": 1,
        "status": "dry_run",
        "workflow": {
            "name": cfg.name,
            "description": cfg.description,
            "config_path": str(cfg.config_path),
            "mode": cfg.mode,
        },
        "store": {
            "path": str(cfg.store_path),
            "backend": cfg.store_backend,
            "schema_version": store_summary.schema_version,
            "table_counts": dict(store_summary.table_counts),
        },
        "selection": {
            "current_state_policy": cfg.current_state_policy,
            "object_filter": {
                "object_ids": list(cfg.object_ids),
                "tags": list(cfg.tags),
            },
            "selected_object_count": selected_object_count,
        },
        "propagation": {
            "family": "OGP",
            "model": "ogp",
            "compatibility_alias_accepted": "sgp4",
            "backend": cfg.propagation_backend,
            "output_frame": cfg.output_frame,
            **({"workers_requested": cfg.propagation_workers} if cfg.propagation_backend == "rust" else {}),
            "start_jd_utc": cfg.start_jd_utc,
            "duration_s": cfg.duration_s,
            "step_s": cfg.step_s,
            "chunk_duration_s": cfg.chunk_duration_s,
            "sample_count_per_object": sample_count_per_object,
            "sample_count_per_chunk": sample_count_per_chunk,
            "chunk_count": chunk_count,
            "estimated_total_samples": estimated_samples,
            "expected_product_count": expected_propagation_chunks,
            "existing_product_count": existing_propagation_products,
            "pending_product_count_estimate": pending_propagation_chunks,
        },
        "screening": {
            "mode": cfg.screening_mode,
            "include_previous_close_pairs": cfg.include_previous_close_pairs,
            "close_pair_lookback_run_id": cfg.close_pair_lookback_run_id,
            "previous_close_pair_count": previous_close_pair_count,
            "estimated_pair_window_count": estimated_pair_windows,
            "existing_screening_window_count": existing_screening_windows,
            "pending_pair_window_count_estimate": pending_pair_windows,
            "existing_candidate_count": existing_conjunction_candidates,
            "total_candidate_count": total_conjunction_candidates,
            "existing_refinement_count": existing_candidate_refinements,
            "pending_refinement_count": pending_candidate_refinements,
            "distance_threshold_km": cfg.distance_threshold_km,
            "promotion_limit": cfg.promotion_limit,
            **(
                {
                    "distance_comparison": {
                        "mode": cfg.screening_distance_comparison,
                        "equivalence": "rounding_level",
                        "provenance": "explicit_scale_screening_config",
                    }
                }
                if cfg.screening_distance_comparison != "exact_sqrt"
                else {}
            ),
        },
        "refinement": {
            "method": "sgp4_tighter_sampled_distance",
            "step_s": cfg.refinement_step_s,
            "half_window_s": cfg.refinement_half_window_s,
            "limit": cfg.refinement_limit,
            "ranking_policy": cfg.ranking_policy,
            "existing_refinement_count": existing_candidate_refinements,
            "pending_refinement_count": pending_candidate_refinements,
        },
        "onp_handoff": {
            "enabled": cfg.hpop_handoff_enabled,
            "output_dir": str(cfg.hpop_handoff_output_dir),
            "force_model_profile": cfg.force_model_profile,
            "limit": cfg.hpop_handoff_limit,
            "case_half_window_s": cfg.hpop_handoff_half_window_s,
            "existing_packet_count": existing_hpop_handoff_packets,
            "pending_packet_count": pending_hpop_handoff_packets,
            "generated_configs": 0,
            "note": "Dry-run only: no ONP configs were generated or validated.",
        },
        "hpop_handoff": {
            "enabled": cfg.hpop_handoff_enabled,
            "output_dir": str(cfg.hpop_handoff_output_dir),
            "force_model_profile": cfg.force_model_profile,
            "limit": cfg.hpop_handoff_limit,
            "case_half_window_s": cfg.hpop_handoff_half_window_s,
            "existing_packet_count": existing_hpop_handoff_packets,
            "pending_packet_count": pending_hpop_handoff_packets,
            "generated_configs": 0,
            "deprecated": True,
            "replacement": "onp_handoff",
            "note": "Compatibility alias; use onp_handoff.",
        },
        "onp_refinement": {
            "enabled": cfg.hpop_handoff_enabled,
            "output_dir": str(cfg.hpop_handoff_output_dir / "onp_refinement"),
            "force_model_profile": cfg.force_model_profile,
            "existing_case_count": existing_onp_refinement_cases,
            "pending_materialization_count": pending_onp_materializations,
            "pending_execution_count": onp_execution["pending_execution_count"],
            "active_execution_count": onp_execution["active_execution_count"],
            "completed_execution_count": onp_execution["completed_execution_count"],
            "retryable_failed_count": onp_execution["retryable_failed_count"],
            "initial_state_frame": "eci",
            "frame_transform": "teme_to_eci_iau80",
            "note": "Validated cases can be resumed through the ONP execution queue.",
        },
        "sensitivity": {
            "enabled": cfg.sensitivity_enabled,
            "campaign_type": cfg.sensitivity_campaign_type,
            "target": cfg.sensitivity_target,
            "uncertainty_interpretation": cfg.sensitivity_uncertainty_interpretation,
            "seed": cfg.sensitivity_seed,
            "trial_count": cfg.sensitivity_trial_count,
            "estimated_trial_count": expected_sensitivity_trials,
            "sweep_mode": cfg.sensitivity_sweep_mode,
            "parameter_count": len(cfg.sensitivity_parameters),
            "derived_parameter_count": len(cfg.sensitivity_derived_parameters),
            "frame": dict(cfg.sensitivity_frame),
            "trial_builder": dict(cfg.sensitivity_trial_builder),
            "output_dir": str(cfg.sensitivity_output_dir),
            "non_claim": "Sensitivity v1 records analyst-defined trial assumptions; it does not claim covariance-derived probability.",
        },
        "monte_carlo": {
            "enabled": cfg.monte_carlo_enabled,
            "campaign_type": cfg.monte_carlo_campaign_type,
            "target": cfg.monte_carlo_target,
            "uncertainty_interpretation": cfg.monte_carlo_uncertainty_interpretation,
            "seed": cfg.monte_carlo_seed,
            "trial_count": cfg.monte_carlo_trial_count,
            "estimated_trial_count": expected_monte_carlo_trials,
            "parameter_count": len(cfg.monte_carlo_parameters),
            "derived_parameter_count": len(cfg.monte_carlo_derived_parameters),
            "frame": dict(cfg.monte_carlo_frame),
            "trial_builder": dict(cfg.monte_carlo_trial_builder),
            "output_dir": str(cfg.monte_carlo_output_dir),
            "non_claim": "Monte Carlo v1 records sampled trial assumptions; it does not generate perturbed SGP4 products yet.",
        },
        "resources": {
            "cpu_count": snapshot.cpu_count,
            "load_1m": snapshot.load_1m,
            "load_per_cpu": snapshot.load_per_cpu,
            "total_memory_mb": snapshot.total_memory_mb,
            "available_memory_mb": snapshot.available_memory_mb,
            "disk_total_mb": _bytes_to_mb(disk.total),
            "disk_free_mb": _bytes_to_mb(disk.free),
            "planner_limits": dict(cfg.planner_limits),
        },
        "invalidation": {
            "open_items": store_summary.open_invalidation_count,
            "by_reason": dict(store_summary.invalidation_by_reason),
        },
        "planned_jobs": planned_jobs,
        "validation_checks": [
            "config_loaded",
            "store_schema_initialized",
            "current_state_policy_recorded",
            "sgp4_near_earth_support_not_yet_evaluated",
            "output_paths_resolved",
            "dry_run_written",
        ],
        "non_claims": [
            "No propagation was executed.",
            "No conjunction candidates were generated.",
            "No ONP scenarios were generated or validated.",
            "This dry-run is planning evidence, not screening evidence.",
        ],
        "warnings": warnings,
        "artifacts": {},
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    workflow = dict(payload.get("workflow", {}) or {})
    propagation = dict(payload.get("propagation", {}) or {})
    selection = dict(payload.get("selection", {}) or {})
    screening = dict(payload.get("screening", {}) or {})
    refinement = dict(payload.get("refinement", {}) or {})
    hpop_handoff = dict(payload.get("hpop_handoff", {}) or {})
    onp_refinement = dict(payload.get("onp_refinement", {}) or {})
    sensitivity = dict(payload.get("sensitivity", {}) or {})
    monte_carlo = dict(payload.get("monte_carlo", {}) or {})
    store = dict(payload.get("store", {}) or {})
    lines = [
        f"# OEL Scale Planner Dry Run: {workflow.get('name', 'scale_workflow')}",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Config: `{workflow.get('config_path')}`",
        f"- Store: `{store.get('path')}`",
        f"- Store schema: `{store.get('schema_version')}`",
        f"- Selected objects: `{selection.get('selected_object_count')}`",
        f"- Propagation: `{propagation.get('model')}` via `{propagation.get('backend')}`",
        f"- Horizon: `{propagation.get('duration_s')}` s at `{propagation.get('step_s')}` s cadence",
        f"- Estimated samples: `{propagation.get('estimated_total_samples')}`",
        f"- Previous close pairs: `{screening.get('previous_close_pair_count')}`",
        f"- Existing screening windows: `{screening.get('existing_screening_window_count')}`",
        f"- Pending pair windows: `{screening.get('pending_pair_window_count_estimate')}`",
        f"- Pending candidate refinements: `{refinement.get('pending_refinement_count')}`",
        f"- Pending ONP handoff packets: `{hpop_handoff.get('pending_packet_count')}`",
        f"- Pending ONP materializations: `{onp_refinement.get('pending_materialization_count')}`",
        f"- Sensitivity trials: `{sensitivity.get('estimated_trial_count')}`",
        f"- Monte Carlo trials: `{monte_carlo.get('estimated_trial_count')}`",
        "",
        "## Planned Jobs",
        "",
    ]
    for job in payload.get("planned_jobs", []) or []:
        lines.append(f"- `{job.get('job_type')}`: `{job.get('count')}` {job.get('status')}")
    warnings = list(payload.get("warnings", []) or [])
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in payload.get("non_claims", []) or [])
    return "\n".join(lines) + "\n"


def resolve_selected_object_ids(cfg: ScalePlannerConfig) -> tuple[str, ...]:
    """Resolve executable current TLE states after applying IDs and tag filters."""

    return tuple(
        state.object_id
        for state in load_current_tle_states(
            cfg.store_path,
            object_ids=cfg.object_ids,
            tags=cfg.tags,
        )
    )


def _warnings(cfg: ScalePlannerConfig, *, selected_object_count: int) -> list[str]:
    warnings: list[str] = []
    if selected_object_count == 0:
        warnings.append("No current catalog objects were available for propagation planning.")
    if cfg.output_frame != "teme":
        warnings.append("Phase 0 recommends native TEME output for OGP screening evidence.")
    if cfg.propagation_backend == "numpy":
        warnings.append("Non-scalar backends are accepted in the dry-run config but are not implemented yet.")
    if cfg.sensitivity_enabled and cfg.sensitivity_uncertainty_interpretation != "analyst_defined_sensitivity":
        warnings.append("Only analyst-defined sensitivity campaigns are supported in the Phase 1F skeleton.")
    if cfg.sensitivity_enabled and not cfg.sensitivity_parameters:
        warnings.append("Sensitivity is enabled but no campaign parameters were provided.")
    if cfg.monte_carlo_enabled and cfg.monte_carlo_trial_count <= 0:
        warnings.append("Monte Carlo is enabled but trial_count is not positive.")
    if cfg.monte_carlo_enabled and not cfg.monte_carlo_parameters:
        warnings.append("Monte Carlo is enabled but no campaign parameters were provided.")
    return warnings


def _expected_propagation_windows(cfg: ScalePlannerConfig) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (window.start_jd_utc, window.end_jd_utc, window.step_s)
        for window in build_propagation_windows(
            start_jd_utc=cfg.start_jd_utc,
            duration_s=cfg.duration_s,
            step_s=cfg.step_s,
            chunk_duration_s=cfg.chunk_duration_s,
        )
    )


def _estimate_sensitivity_trial_count(cfg: ScalePlannerConfig) -> int:
    if not cfg.sensitivity_enabled:
        return 0
    if cfg.sensitivity_trial_count > 0:
        return cfg.sensitivity_trial_count
    counts: list[int] = []
    for parameter in cfg.sensitivity_parameters:
        mode = str(parameter.get("mode", "fixed") or "fixed").strip().lower()
        if mode == "values":
            counts.append(len(parameter.get("values", []) or []))
        elif mode == "linspace":
            counts.append(int(max(int(parameter.get("count", 0) or 0), 0)))
        else:
            counts.append(1)
    if not counts:
        return 0
    if cfg.sensitivity_sweep_mode == "grid":
        count = 1
        for item in counts:
            count *= item
        return int(count)
    return int(max(counts))


def _dict_section(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected a list of strings.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _truthy(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _validated_planner_limits(planner_cfg: dict[str, Any]) -> dict[str, float | None]:
    limits: dict[str, float | None] = {}
    for name in ("max_runtime_s", "max_memory_mb", "max_storage_mb"):
        value = planner_cfg.get(name)
        if value is None:
            limits[name] = None
            continue
        out = float(value)
        if not math.isfinite(out) or out <= 0.0:
            raise ValueError(f"scale_catalog.planner.{name} must be a finite positive number or null.")
        limits[name] = out
    return limits


def _positive_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"scale_catalog.propagation.{name} must be positive and finite.")
    return out


def _finite_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"scale_catalog.propagation.{name} must be finite.")
    return out


def _bytes_to_mb(value: int) -> float:
    return round(float(value) / (1024.0 * 1024.0), 3)


def _resolve_output_file(path_text: str | Path, *, path_policy: ConfigPathPolicy, purpose: str) -> Path:
    raw = Path(str(path_text or "").strip())
    if not raw.name:
        raise ValueError(f"{purpose} must include a file name.")
    parent_text = str(raw.parent) if str(raw.parent) not in {"", "."} else "outputs/scale"
    parent = path_policy.resolve_output_dir(parent_text, purpose=f"{purpose}.parent")
    return parent / raw.name
