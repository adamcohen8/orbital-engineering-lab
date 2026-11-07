from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim import SimulationConfig, SimulationSession


def _config(tmp_path: Path, *, duration_s: float, dt_s: float) -> SimulationConfig:
    return SimulationConfig.from_dict(
        {
            "scenario_name": "runtime_time_progress_regression",
            "metadata": {"seed": 7},
            "objects": {
                "probe": {
                    "enabled": True,
                    "role": "target",
                    "kind": "satellite",
                    "specs": {"mass_kg": 100.0},
                    "initial_state": {
                        "position_eci_km": [7000.0, 0.0, 0.0],
                        "velocity_eci_km_s": [0.0, 7.5, 0.0],
                    },
                }
            },
            "simulator": {
                "initial_jd_utc": 2461254.5,
                "duration_s": duration_s,
                "dt_s": dt_s,
                "dynamics": {
                    "orbit": {"model": "two_body", "orbit_substep_s": dt_s},
                    "attitude": {"enabled": False, "attitude_substep_s": dt_s},
                    "rocket": {"enabled": False},
                },
                "execution": {
                    "runtime_profiler": {"enabled": False},
                    "object_parallelism": {"enabled": False, "backend": "serial"},
                },
                "termination": {"earth_impact_enabled": False},
            },
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
                "plots": {"enabled": False},
                "animations": {"enabled": False},
            },
        }
    )


def test_nonadvancing_step_rejected_before_history_mutation(tmp_path: Path) -> None:
    session = SimulationSession.from_config(_config(tmp_path, duration_s=2.0, dt_s=1.0))
    first = session.step()
    assert first.time_s == pytest.approx(1.0)
    assert session._engine is not None
    engine = session._engine
    before_index = int(engine.current_index)
    before_times = engine.t_s.copy()
    before_sample_offset = int(engine.sample_offset)
    before_capacity = int(engine.n)

    with pytest.raises(ValueError, match="too small to advance"):
        session.step(dt_s=float(np.nextafter(0.0, 1.0)))

    assert int(engine.current_index) == before_index
    assert int(engine.sample_offset) == before_sample_offset
    assert int(engine.n) == before_capacity
    np.testing.assert_array_equal(engine.t_s, before_times)
    assert session._step_index == before_index
    assert session.done is False


def test_positive_subnanosecond_duration_executes_a_step(tmp_path: Path) -> None:
    duration_s = 1.0e-10
    session = SimulationSession.from_config(_config(tmp_path, duration_s=duration_s, dt_s=duration_s))

    result = session.run()

    np.testing.assert_array_equal(result.time_s, np.array([0.0, duration_s]))
    assert result.num_steps == 2
    assert session.done is True


def test_dynamic_history_compaction_refreshes_retained_write_index(tmp_path: Path) -> None:
    session = SimulationSession.from_config(
        _config(tmp_path, duration_s=6.0, dt_s=1.0),
        history_mode="dynamic",
        initial_history_capacity=4,
        max_history_samples=4,
    )

    for _ in range(6):
        snapshot = session.step()

    assert snapshot.step_index == 6
    assert snapshot.time_s == pytest.approx(6.0)
    assert session._engine is not None
    engine = session._engine
    assert engine.allocated_history_samples == 4
    assert engine.sample_offset > 0
    assert engine.retained_end_step == 6
    assert engine.current_index < engine.n
