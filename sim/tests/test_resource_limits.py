from __future__ import annotations

from pathlib import Path

import pytest

from sim.config import scenario_config_from_dict
from sim.execution import create_single_run_engine
from sim.resource_limits import (
    ResourceGovernor,
    ResourcePressureError,
    ResourceSnapshot,
    SimulationMemoryBudgetError,
    _memory_bytes_to_mb,
    _parse_macos_memory_pressure,
    _parse_macos_vm_stat_mb,
    apply_resource_profile_to_config_dict,
    current_resource_snapshot,
    estimate_history_memory_from_config,
    estimate_resource_requirements,
)


def _single_target_config(output_dir: Path, *, duration_s: float = 1000.0) -> dict:
    return {
        "scenario_name": "memory_guard_test",
        "objects": {
            "target": {
                "enabled": True,
                "specs": {"mass_kg": 100.0},
                "initial_state": {
                    "position_eci_km": [7000.0, 0.0, 0.0],
                    "velocity_eci_km_s": [0.0, 7.5, 0.0],
                },
            },
        },
        "simulator": {
            "duration_s": duration_s,
            "dt_s": 1.0,
            "termination": {"earth_impact_enabled": False},
            "dynamics": {"attitude": {"enabled": False}},
        },
        "outputs": {
            "output_dir": str(output_dir),
            "mode": "save",
            "stats": {"print_summary": False, "save_json": False, "save_full_log": False},
            "plots": {"enabled": False},
            "animations": {"enabled": False},
        },
    }


def test_history_memory_guard_fails_before_output_directory_creation(tmp_path: Path) -> None:
    outdir = tmp_path / "guarded"
    root = _single_target_config(outdir)
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 0.01}
    cfg = scenario_config_from_dict(root)

    with pytest.raises(SimulationMemoryBudgetError, match="Estimated simulation history memory"):
        create_single_run_engine(cfg)

    assert not outdir.exists()


def test_history_memory_guard_uses_caller_cap_even_if_config_requests_more(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _single_target_config(tmp_path / "guarded")
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 1000.0}
    cfg = scenario_config_from_dict(root)
    monkeypatch.setenv("OEL_MAX_HISTORY_MEMORY_MB", "0.01")

    with pytest.raises(SimulationMemoryBudgetError, match="limit=0.01 MB"):
        create_single_run_engine(cfg)


def test_history_memory_guard_allows_run_under_budget(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "ok", duration_s=2.0)
    root["outputs"]["resource_limits"] = {"max_history_memory_mb": 10.0}
    cfg = scenario_config_from_dict(root)

    engine = create_single_run_engine(cfg)

    assert engine.history_memory_estimate.estimated_peak_mb < 10.0


def test_config_history_estimate_matches_engine_allocation_contract(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "rich", duration_s=20.0)
    root["simulator"]["dynamics"]["attitude"]["enabled"] = True
    root["objects"]["target"]["knowledge"] = {"targets": ["target"]}
    root["outputs"]["stats"]["controller_debug"] = True
    cfg = scenario_config_from_dict(root)

    preflight = estimate_history_memory_from_config(cfg)
    engine = create_single_run_engine(cfg)

    assert preflight == engine.history_memory_estimate


def test_laptop_safe_profile_rewrites_batch_config(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=10.0)
    root["analysis"] = {
        "enabled": True,
        "study_type": "monte_carlo",
        "execution": {"parallel_enabled": True, "parallel_workers": 8},
        "monte_carlo": {"iterations": 4, "variations": []},
    }
    root["outputs"]["plots"] = {"enabled": True}

    profiled = apply_resource_profile_to_config_dict(root, "laptop-safe")

    assert profiled["analysis"]["execution"]["parallel_enabled"] is False
    assert profiled["analysis"]["execution"]["parallel_workers"] == 1
    assert profiled["outputs"]["plots"]["enabled"] is False
    assert profiled["outputs"]["monte_carlo"]["checkpoint_enabled"] is True
    assert profiled["outputs"]["resource_limits"]["min_available_memory_mb"] == 512.0
    assert profiled["outputs"]["resource_limits"]["hard_min_available_memory_mb"] == 256.0
    assert profiled["simulator"]["resource_profile"] == "laptop-safe"


def test_resource_estimate_reports_effective_batch_shape(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=12.0)
    root["analysis"] = {
        "enabled": True,
        "study_type": "monte_carlo",
        "execution": {"parallel_enabled": True, "parallel_workers": 2},
        "monte_carlo": {"iterations": 3, "variations": []},
    }
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.study_type == "monte_carlo"
    assert estimate.runs == 3
    assert estimate.steps_per_run == 12
    assert estimate.effective_workers == 2


def test_resource_estimate_uses_configured_load_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=12.0)
    root["analysis"] = {
        "enabled": True,
        "study_type": "monte_carlo",
        "execution": {"parallel_enabled": True, "parallel_workers": 2},
        "monte_carlo": {"iterations": 3, "variations": []},
    }
    root["outputs"]["resource_limits"] = {
        "throttle_enabled": True,
        "max_load_per_cpu": 0.1,
    }
    cfg = scenario_config_from_dict(root)
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=8.0,
        total_memory_mb=8192.0,
        available_memory_mb=4096.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    estimate = estimate_resource_requirements(cfg)
    governor = ResourceGovernor(cfg)

    assert estimate.risk == "heavy"
    assert "current system load is elevated for a parallel launch" in estimate.notes
    assert governor.pressure_reasons(snapshot) == ["load/core 1.00 > 0.10"]


@pytest.mark.parametrize("field", [
    "min_available_memory_mb",
    "hard_min_available_memory_mb",
    "max_load_per_cpu",
])
def test_resource_governor_rejects_negative_pressure_thresholds(tmp_path: Path, field: str) -> None:
    root = _single_target_config(tmp_path / field, duration_s=2.0)
    root["outputs"]["resource_limits"] = {field: -1.0}
    cfg = scenario_config_from_dict(root)

    with pytest.raises(ValueError, match="nonnegative"):
        ResourceGovernor(cfg)


@pytest.mark.parametrize("field", [
    "min_available_memory_mb",
    "hard_min_available_memory_mb",
    "max_load_per_cpu",
])
def test_resource_governor_rejects_boolean_pressure_thresholds(tmp_path: Path, field: str) -> None:
    root = _single_target_config(tmp_path / field, duration_s=2.0)
    root["outputs"]["resource_limits"] = {field: False}
    cfg = scenario_config_from_dict(root)

    with pytest.raises(ValueError, match="not boolean"):
        ResourceGovernor(cfg)


def test_resource_governor_honors_zero_max_wait(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "zero_wait", duration_s=2.0)
    root["outputs"]["resource_limits"] = {"resource_max_wait_s": 0.0}
    cfg = scenario_config_from_dict(root)

    governor = ResourceGovernor(cfg)

    assert governor.max_wait_s == 0.0


def test_resource_governor_rejects_zero_pause_seconds(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "zero_pause", duration_s=2.0)
    root["outputs"]["resource_limits"] = {"resource_pause_seconds": 0.0}
    cfg = scenario_config_from_dict(root)

    with pytest.raises(ValueError, match="positive and finite"):
        ResourceGovernor(cfg)


def test_resource_estimate_reports_sensitivity_shape(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "sensitivity", duration_s=12.0)
    root["analysis"] = {
        "enabled": True,
        "study_type": "sensitivity",
        "execution": {"parallel_enabled": True, "parallel_workers": 3},
        "sensitivity": {
            "method": "two_parameter_grid",
            "parameters": [
                {"parameter_path": "objects.target.specs.mass_kg", "values": [90.0, 100.0]},
                {"parameter_path": "objects.target.specs.drag_area_m2", "values": [0.8, 1.0, 1.2]},
            ],
        },
    }
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.study_type == "sensitivity"
    assert estimate.runs == 6
    assert estimate.requested_workers == 3
    assert estimate.effective_workers == 3


def test_resource_governor_honors_off_profile(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=2.0)
    root["simulator"]["resource_profile"] = "off"
    cfg = scenario_config_from_dict(root)

    governor = ResourceGovernor(cfg)

    assert governor.enabled is False


def test_simulator_resource_profile_is_canonical_over_legacy_outputs_profile(tmp_path: Path) -> None:
    root = _single_target_config(tmp_path / "mc", duration_s=2.0)
    root["simulator"]["resource_profile"] = "laptop-safe"
    root["outputs"]["resource_limits"] = {"resource_profile": "off"}
    cfg = scenario_config_from_dict(root)

    estimate = estimate_resource_requirements(cfg)
    governor = ResourceGovernor(cfg)

    assert estimate.profile == "laptop-safe"
    assert governor.profile.name == "laptop-safe"


@pytest.mark.parametrize(
    ("available_mb", "expected_risk", "expected_action"),
    [
        (700.0, "safe", "proceed"),
        (500.0, "heavy", "advisory"),
        (200.0, "unsafe", "refuse"),
    ],
)
def test_laptop_safe_uses_projected_headroom_for_small_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_mb: float,
    expected_risk: str,
    expected_action: str,
) -> None:
    root = apply_resource_profile_to_config_dict(
        _single_target_config(tmp_path / "small", duration_s=2.0),
        "laptop-safe",
    )
    cfg = scenario_config_from_dict(root)
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=8192.0,
        available_memory_mb=available_mb,
        memory_pressure_free_percent=40.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.risk == expected_risk
    assert estimate.action == expected_action


def test_laptop_safe_refuses_large_increment_despite_nominal_current_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _single_target_config(tmp_path / "large", duration_s=100_000.0)
    root["outputs"]["stats"]["controller_debug"] = True
    cfg = scenario_config_from_dict(apply_resource_profile_to_config_dict(root, "laptop-safe"))
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=8192.0,
        available_memory_mb=700.0,
        memory_pressure_free_percent=40.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.estimated_incremental_memory_mb > estimate.current_available_memory_mb
    assert estimate.risk == "unsafe"


def test_serial_resource_wait_ignores_trailing_load_average(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = scenario_config_from_dict(
        apply_resource_profile_to_config_dict(_single_target_config(tmp_path / "serial", duration_s=2.0), "laptop-safe")
    )
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=24.0,
        total_memory_mb=8192.0,
        available_memory_mb=4000.0,
        memory_pressure_free_percent=40.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)
    governor = ResourceGovernor(cfg)

    assert governor.pressure_reasons(snapshot, include_load=False) == []
    assert governor.pressure_reasons(snapshot, include_load=True) == ["load/core 3.00 > 1.25"]


def test_macos_critical_pressure_is_unsafe_even_with_nominal_available_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = scenario_config_from_dict(
        apply_resource_profile_to_config_dict(_single_target_config(tmp_path / "critical", duration_s=2.0), "laptop-safe")
    )
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=8192.0,
        available_memory_mb=4000.0,
        memory_pressure_free_percent=4.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    assert estimate_resource_requirements(cfg).risk == "unsafe"
    with pytest.raises(ResourcePressureError, match="critical memory pressure"):
        ResourceGovernor(cfg).assert_safe_to_start()


def test_guarded_profile_fails_closed_when_memory_telemetry_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = scenario_config_from_dict(
        apply_resource_profile_to_config_dict(_single_target_config(tmp_path / "unknown", duration_s=2.0), "laptop-safe")
    )
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=None,
        available_memory_mb=None,
        memory_pressure_free_percent=None,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    estimate = estimate_resource_requirements(cfg)

    assert estimate.risk == "unsafe"
    assert "memory telemetry is unavailable for the configured safety floor" in estimate.notes
    with pytest.raises(ResourcePressureError, match="Memory telemetry is unavailable"):
        ResourceGovernor(cfg).assert_safe_to_start()


def test_sensitivity_resource_estimate_scales_with_retained_payload_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=16384.0,
        available_memory_mb=12000.0,
        memory_pressure_free_percent=50.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    def _estimate(values: list[float]):
        root = _single_target_config(tmp_path / f"sensitivity-{len(values)}", duration_s=12.0)
        root["analysis"] = {
            "enabled": True,
            "study_type": "sensitivity",
            "execution": {"parallel_enabled": False},
            "sensitivity": {
                "method": "one_at_a_time",
                "parameters": [
                    {"parameter_path": "objects.target.specs.mass_kg", "values": values},
                ],
            },
        }
        return estimate_resource_requirements(scenario_config_from_dict(root))

    two_runs = _estimate([90.0, 110.0])
    five_runs = _estimate([80.0, 90.0, 100.0, 110.0, 120.0])

    assert five_runs.estimated_incremental_memory_mb > two_runs.estimated_incremental_memory_mb
    assert "full run payloads retained for 2 batch runs" in two_runs.notes
    assert "full run payloads retained for 5 batch runs" in five_runs.notes


def test_monte_carlo_payload_gate_adds_iteration_scaled_retention_to_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=16384.0,
        available_memory_mb=12000.0,
        memory_pressure_free_percent=50.0,
    )
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: snapshot)

    def _estimate(iterations: int, *, payload_gate: bool):
        root = _single_target_config(tmp_path / f"mc-{iterations}-{payload_gate}", duration_s=12.0)
        root["analysis"] = {
            "enabled": True,
            "study_type": "monte_carlo",
            "execution": {"parallel_enabled": False},
            "monte_carlo": {"iterations": iterations, "variations": []},
        }
        if payload_gate:
            root["outputs"]["monte_carlo"] = {
                "gates": {
                    "metric_gates": [
                        {"metric": "payload.truth_by_object.target", "op": "!=", "value": None},
                    ],
                },
            }
        return estimate_resource_requirements(scenario_config_from_dict(root))

    two_runs = _estimate(2, payload_gate=True)
    five_runs = _estimate(5, payload_gate=True)
    five_runs_without_payloads = _estimate(5, payload_gate=False)

    assert five_runs.estimated_incremental_memory_mb > two_runs.estimated_incremental_memory_mb
    assert five_runs.estimated_incremental_memory_mb > five_runs_without_payloads.estimated_incremental_memory_mb
    assert "full run payloads retained for 5 batch runs" in five_runs.notes


def test_resource_governor_reports_wait_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = scenario_config_from_dict(
        apply_resource_profile_to_config_dict(_single_target_config(tmp_path / "wait", duration_s=2.0), "laptop-safe")
    )
    high = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=8192.0,
        available_memory_mb=700.0,
        memory_pressure_free_percent=40.0,
    )
    low = ResourceSnapshot(
        timestamp_s=0.0,
        cpu_count=8,
        load_1m=0.0,
        total_memory_mb=8192.0,
        available_memory_mb=400.0,
        memory_pressure_free_percent=40.0,
    )
    snapshots = iter([high, low, high])
    monkeypatch.setattr("sim.resource_limits.current_resource_snapshot", lambda: next(snapshots))
    monkeypatch.setattr("sim.resource_limits.time.sleep", lambda _seconds: None)
    messages: list[str] = []
    governor = ResourceGovernor(cfg, emit=messages.append)

    governor.wait_for_capacity(context="test iteration", include_load=False)

    assert governor.wait_event_count == 1
    assert governor.total_wait_s >= 0.0
    assert messages[0].startswith("Throttling test iteration:")
    assert messages[-1].startswith("Resuming test iteration after ")


def test_macos_vm_stat_parser_counts_reclaimable_inactive_memory() -> None:
    text = """
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                7100.
Pages active:                            130919.
Pages inactive:                          127528.
Pages speculative:                         2780.
Pages wired down:                        145265.
Pages purgeable:                           4311.
Pages occupied by compressor:             74159.
""".strip()

    total_mb, available_mb = _parse_macos_vm_stat_mb(text)

    free_only_mb = (7100 + 2780) * 16384 / (1024 * 1024)
    assert available_mb is not None
    assert total_mb is not None
    assert available_mb > free_only_mb * 10


def test_macos_memory_pressure_parser_reads_total_and_free_percentage() -> None:
    text = """
The system has 8589934592 (524288 pages with a page size of 16384).
System-wide memory free percentage: 44%
""".strip()

    total_mb, free_percent = _parse_macos_memory_pressure(text)

    assert total_mb == 8192.0
    assert free_percent == 44.0


def test_windows_memory_bytes_convert_to_mb() -> None:
    total_mb, available_mb = _memory_bytes_to_mb(16 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024)

    assert total_mb == 16384.0
    assert available_mb == 5120.0


def test_resource_snapshot_uses_windows_memory_reader_when_unix_readers_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sim.resource_limits._read_proc_mem_available_mb", lambda: (None, None))
    monkeypatch.setattr("sim.resource_limits._read_macos_memory_mb", lambda: (None, None))
    monkeypatch.setattr("sim.resource_limits._read_macos_memory_pressure", lambda: (None, None))
    monkeypatch.setattr("sim.resource_limits._read_windows_memory_mb", lambda: (8192.0, 4096.0))

    snapshot = current_resource_snapshot()

    assert snapshot.total_memory_mb == 8192.0
    assert snapshot.available_memory_mb == 4096.0
