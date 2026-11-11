# ruff: noqa: F401,I001
from __future__ import annotations

import importlib
import logging
import math
import multiprocessing as mp
import os
import queue as queue_mod
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from sim.config import (
    configured_objects,
    load_simulation_yaml,
    scenario_config_from_dict,
)
from sim.config.plugin_validation import validate_scenario_plugins
from sim.controller_lab.metrics import evaluate_metric
from sim.controller_lab.models import (
    ControllerBenchCase,
    ControllerBenchConfig,
    ControllerBenchEnvelopeConfig,
    ControllerBenchEnvelopeVariation,
    ControllerBenchEngineeringConfig,
    ControllerBenchMetric,
    ControllerBenchObjective,
    ControllerBenchOptimizationConfig,
    ControllerBenchOptimizationCost,
    ControllerBenchOptimizationParameter,
    ControllerBenchPassCriterion,
    ControllerBenchTarget,
    ControllerVariant,
    ControllerVariantOverride,
    ControllerVariantPointer,
)
from sim.controller_lab.plotting import render_controller_bench_visualizations, write_linear_feedback_diagnostics
from sim.controller_lab.reporting import write_controller_bench_reports
from sim.execution.hierarchical import (
    apply_hierarchical_worker_env,
    plan_hierarchical_execution,
    restore_hierarchical_worker_env,
    restrictive_profile_for_task_roots,
)
from sim.execution.parameter_paths import object_synced_parameter_paths, set_parameter_path_value
from sim.execution.study import analysis_study_type
from sim.execution.workers import restore_env_vars, set_parallel_worker_thread_limits
from sim.pro_features import FEATURE_CONTROLLER_BENCH, require_pro_feature
from sim.single_run import _run_single_config
from sim.utils.parallel import (
    format_parallel_fallback_reason,
    initialize_worker_progress_queue,
    iter_bounded_futures,
    worker_progress_queue,
)

logger = logging.getLogger(__name__)


def _metric_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"

__all__ = [name for name in globals() if not name.startswith("__")]
