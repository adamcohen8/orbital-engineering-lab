from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from sim.config import load_simulation_yaml
from sim.resource_limits import (
    _planned_step_count,
    _simulation_completion_tolerance_s,
    estimate_history_memory_from_config,
)
from sim.single_run import _SingleRunEngine

ROOT = Path(__file__).resolve().parents[1]


def test_nonintegral_duration_allocates_partial_final_step_and_progress_budget() -> None:
    config = load_simulation_yaml(ROOT.parent / "configs" / "automation_smoke.yaml")
    simulator = replace(config.simulator, duration_s=2.5, dt_s=1.0)
    config = replace(config, simulator=simulator)

    engine = _SingleRunEngine(config)

    assert engine.planned_samples == 4
    assert engine.total_steps == 3


def test_completion_tolerance_is_local_ulp_for_long_duration() -> None:
    duration_s = 1.0e12

    assert _simulation_completion_tolerance_s(duration_s) == math.ulp(duration_s)
    assert _simulation_completion_tolerance_s(duration_s) < 1.0


def test_long_duration_done_boundary_does_not_consume_scale_relative_seconds() -> None:
    duration_s = 1.0e12
    config = load_simulation_yaml(ROOT.parent / "configs" / "automation_smoke.yaml")
    simulator = replace(config.simulator, duration_s=duration_s, dt_s=1.0)
    config = replace(config, simulator=simulator)
    engine = _SingleRunEngine(
        config,
        history_mode="dynamic",
        initial_history_capacity=2,
        max_history_samples=2,
    )

    engine.t_s[0] = duration_s - 1.0
    assert engine.done is False
    engine.t_s[0] = duration_s - math.ulp(duration_s)
    assert engine.done is True


@pytest.mark.parametrize(
    ("duration_s", "dt_s"),
    [
        (1.0e308, 1.0e-308),  # ratio overflows to infinity
        (1.0e20, 1.0),  # finite ratio cannot be an array dimension
    ],
)
def test_unrepresentable_step_ratio_is_rejected_before_history_allocation(
    duration_s: float,
    dt_s: float,
) -> None:
    with pytest.raises(ValueError, match="duration_s/dt_s ratio"):
        _planned_step_count(duration_s, dt_s)

    config = load_simulation_yaml(ROOT.parent / "configs" / "automation_smoke.yaml")
    simulator = replace(config.simulator, duration_s=duration_s, dt_s=dt_s)
    config = replace(config, simulator=simulator)
    with pytest.raises(ValueError, match="duration_s/dt_s ratio"):
        _SingleRunEngine(config)
    with pytest.raises(ValueError, match="duration_s/dt_s ratio"):
        estimate_history_memory_from_config(config)


def test_config_validation_reports_ratio_overflow_as_timing_error() -> None:
    config = load_simulation_yaml(ROOT.parent / "configs" / "automation_smoke.yaml")
    root = config.to_dict()
    root["simulator"]["duration_s"] = 1.0e308
    root["simulator"]["dt_s"] = 1.0e-308

    from sim.config import scenario_config_from_dict

    with pytest.raises(ValueError, match="ratio is not representable"):
        scenario_config_from_dict(root)
