# ruff: noqa: F401,I001
from __future__ import annotations

import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import CancelledError
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from sim.config import (
    SimulationScenarioConfig,
    iter_object_sections,
    load_simulation_yaml,
    scenario_config_from_dict,
    validate_scenario_plugins,
)
from sim.execution.hierarchical import (
    apply_hierarchical_worker_env,
    plan_hierarchical_execution,
    restore_hierarchical_worker_env,
    restrictive_profile_for_task_roots,
)
from sim.execution.metrics import (
    closest_approach_from_run_payload,
    relative_motion_summary_from_run_payload,
    relative_range_series_from_run_payload,
)
from sim.execution.monte_carlo_support import assess_mc_run, safe_float, satellite_initial_delta_v_budget_m_s
from sim.execution.parameter_paths import set_parameter_path_value
from sim.execution.workers import restore_env_vars, run_mc_iteration_from_dict, set_parallel_worker_thread_limits
from sim.pro_features import FEATURE_CAMPAIGNS, require_pro_feature
from sim.reporting.monte_carlo_streaming import MonteCarloRelativeRangePlotWriter
from sim.resource_limits import ResourceGovernor, ResourcePressureError, checkpoint_enabled
from sim.security import ConfigPathPolicy
from sim.single_run import _run_single_config
from sim.utils.io import write_json
from sim.utils.parallel import (
    _drain_parallel_progress_events as _drain_parallel_progress_events,
    _raise_if_cancellation as _raise_if_cancellation,
    abort_process_pool,
    format_parallel_fallback_reason,
    initialize_worker_progress_queue,
    iter_bounded_futures,
    worker_progress_queue,
)

StepCallback = Callable[[int, int], None]
BatchCallback = Callable[[int, int], None]
BatchProgressCallback = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)

_ORIGINAL_CLOSEST_APPROACH = closest_approach_from_run_payload
_ORIGINAL_RELATIVE_RANGE_SERIES = relative_range_series_from_run_payload
_ORIGINAL_RUN_SINGLE_CONFIG = _run_single_config


def _compat_campaign_metric(name, original, payload):
    facade = sys.modules.get("sim.execution.campaigns")
    current = getattr(facade, name, original)
    if current is not original:
        return current(payload)
    return original(payload)


def _compat_closest_approach(payload):
    return _compat_campaign_metric("closest_approach_from_run_payload", _ORIGINAL_CLOSEST_APPROACH, payload)


def _compat_relative_range_series(payload):
    return _compat_campaign_metric(
        "relative_range_series_from_run_payload", _ORIGINAL_RELATIVE_RANGE_SERIES, payload
    )


def _compat_run_single_config(*args, **kwargs):
    facade = sys.modules.get("sim.execution.campaigns")
    current = getattr(facade, "_run_single_config", _ORIGINAL_RUN_SINGLE_CONFIG)
    if current is not _ORIGINAL_RUN_SINGLE_CONFIG:
        return current(*args, **kwargs)
    return _ORIGINAL_RUN_SINGLE_CONFIG(*args, **kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value

def _config_fingerprint(config_dict: dict[str, Any]) -> str:
    from sim.execution.provenance import external_physics_input_digests, runtime_implementation_digest

    payload = json.dumps(
        {
            "config": _json_safe(config_dict),
            "runtime_implementation_digest": runtime_implementation_digest(),
            "external_physics_input_digests": external_physics_input_digests(config_dict),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

__all__ = [name for name in globals() if not name.startswith("__")]
