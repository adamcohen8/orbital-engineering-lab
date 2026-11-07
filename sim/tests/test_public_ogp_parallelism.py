"""Public OGP threads preserve one scenario; ONP workers remain entitled."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from sim.config import scenario_config_from_dict
from sim.single_run import _SingleRunEngine

ROOT = Path(__file__).resolve().parents[2]


def _raw(tmp_path):
    raw = yaml.safe_load((ROOT / "examples/configs/public_ogp_parallel.yaml").read_text())
    raw["outputs"]["output_dir"] = str(tmp_path)
    return raw


def _native():
    from sim.rust_ogp_backend import _extension

    try:
        native = _extension()
    except RuntimeError:
        pytest.skip("requires optional Rust OGP build")
    if not hasattr(native, "ogp_propagate_batch_parallel_bytes"):
        pytest.skip("requires parallel OGP batch binding")
    return native


def _deny(*args, **kwargs):
    raise ImportError("Pro feature unavailable")


def _finish(engine):
    while not engine.done:
        engine.step()
    return engine


def test_public_ogp_matches_serial_history_and_callbacks_without_pro(tmp_path, monkeypatch):
    _native()
    monkeypatch.setattr("sim.execution.object_step_coordinator.require_pro_feature", _deny)
    raw = _raw(tmp_path)
    callbacks = []
    parallel = _finish(_SingleRunEngine(
        scenario_config_from_dict(deepcopy(raw)), step_callback=lambda step, _: callbacks.append(step),
    ))
    raw["simulator"]["execution"]["ogp_parallelism"]["enabled"] = False
    serial = _finish(_SingleRunEngine(scenario_config_from_dict(raw)))
    np.testing.assert_array_equal(parallel.t_s, serial.t_s)
    for oid in parallel.agents:
        np.testing.assert_array_equal(parallel.truth_hist[oid], serial.truth_hist[oid])
    assert callbacks == list(range(parallel.current_index + 1))
    plan = parallel.object_execution_plan
    assert plan["parallel_scope"] == "ogp_history_preparation"
    assert plan["ogp_batches"] == 1
    assert 1 <= plan["selected_workers"] <= 2
    # Native CPU admission is authoritative; don't claim thread use on a
    # single-CPU machine merely because workers=2 was requested.
    assert plan["selected_backend"] == ("native_threads" if plan["selected_workers"] > 1 else "serial")


@pytest.mark.parametrize("profile,policy", [("laptop-safe", "configured"), ("standard", "serial")])
def test_public_ogp_respects_serial_resource_policy(tmp_path, profile, policy):
    _native()
    raw = _raw(tmp_path)
    raw["simulator"].update(resource_profile=profile, duration_s=60.0)
    raw["simulator"]["execution"]["policy"] = policy
    engine = _finish(_SingleRunEngine(scenario_config_from_dict(raw)))
    assert engine.object_execution_plan["selected_workers"] == 1
    assert engine.object_execution_plan["selected_backend"] == "serial"


def test_onp_object_workers_still_require_pro(tmp_path, monkeypatch):
    monkeypatch.setattr("sim.execution.object_step_coordinator.require_pro_feature", _deny)
    raw = _raw(tmp_path)
    for obj in raw["objects"].values():
        obj["propagation_method"] = "special"
        obj.pop("general")
        obj["initial_state"] = {"position_eci_km": [7000.0, 0.0, 0.0], "velocity_eci_km_s": [0.0, 7.5, 0.0]}
    raw["simulator"]["execution"] = {"policy": "parallel", "object_parallelism": {"workers": 2}}
    with pytest.raises(ImportError, match="Pro feature unavailable"):
        _SingleRunEngine(scenario_config_from_dict(raw))


@pytest.mark.parametrize("option,value", [("workers", -1), ("workers", True), ("enabled", "yes"), ("typo", 2)])
def test_ogp_parallel_options_fail_closed(tmp_path, option, value):
    raw = _raw(tmp_path)
    raw["simulator"]["execution"]["ogp_parallelism"][option] = value
    with pytest.raises(ValueError, match="ogp_parallelism"):
        scenario_config_from_dict(raw)


def test_ogp_parallelism_cannot_enable_private_object_workers(tmp_path):
    raw = _raw(tmp_path)
    raw["simulator"]["execution"]["policy"] = "parallel"
    with pytest.raises(ValueError, match="cannot be combined"):
        scenario_config_from_dict(raw)


def test_ogp_only_extension_loader(tmp_path, monkeypatch):
    import sim.rust_ogp_backend as backend

    sentinel = object()

    def importer(name):
        if name == "oel_rust_ogp":
            return sentinel
        raise ImportError(name)

    monkeypatch.setattr(backend, "import_module", importer)
    assert backend._extension() is sentinel


def test_mixed_onp_stays_serial_and_cadence_changes_preserve_ogp(tmp_path, monkeypatch):
    _native()
    monkeypatch.setattr("sim.execution.object_step_coordinator.require_pro_feature", _deny)
    raw = _raw(tmp_path)
    raw["objects"]["onp"] = {
        "kind": "satellite", "runtime_profile": "trajectory_only",
        "propagation_method": "special",
        "initial_state": {"position_eci_km": [7000.0, 0.0, 0.0], "velocity_eci_km_s": [0.0, 7.5, 0.0]},
    }
    parallel = _SingleRunEngine(scenario_config_from_dict(deepcopy(raw)))
    raw["simulator"]["execution"]["ogp_parallelism"]["enabled"] = False
    serial = _SingleRunEngine(scenario_config_from_dict(raw))
    for engine in (parallel, serial):
        for _ in range(260):
            engine.step(dt_s=30.0)
        for _ in range(3):
            engine.step(dt_s=15.0)
    for oid in parallel.agents:
        np.testing.assert_array_equal(parallel.truth_hist[oid], serial.truth_hist[oid])
    assert parallel.object_step_executor.backend_name == "serial"
    assert parallel.object_step_executor.max_workers == 1
    assert parallel.object_execution_plan["ogp_batches"] >= 3


@pytest.mark.parametrize("failure", ["sparse", "runtime"])
def test_future_ogp_errors_do_not_preempt_current_samples(tmp_path, monkeypatch, failure):
    _native()
    import sim.execution.ogp_history as history

    raw = _raw(tmp_path)
    raw["simulator"]["duration_s"] = 120.0
    propagate = history.propagate_batch_with_telemetry

    def future_error(*args, **kwargs):
        if failure == "runtime":
            raise RuntimeError("speculative future failure")
        positions, velocities, errors, telemetry = propagate(*args, **kwargs)
        errors[0, -1] = "speculative future failure"
        return positions, velocities, errors, telemetry

    monkeypatch.setattr(history, "propagate_batch_with_telemetry", future_error)
    parallel = _finish(_SingleRunEngine(scenario_config_from_dict(deepcopy(raw))))
    raw["simulator"]["execution"]["ogp_parallelism"]["enabled"] = False
    serial = _finish(_SingleRunEngine(scenario_config_from_dict(raw)))
    for oid in parallel.agents:
        np.testing.assert_array_equal(parallel.truth_hist[oid], serial.truth_hist[oid])
