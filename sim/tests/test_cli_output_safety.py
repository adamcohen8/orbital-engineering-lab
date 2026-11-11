from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import run_simulation


def _cfg(output_dir, *, duration_s: float = 10.0, dt_s: float = 1.0):
    return SimpleNamespace(
        outputs=SimpleNamespace(output_dir=str(output_dir)),
        simulator=SimpleNamespace(duration_s=duration_s, dt_s=dt_s),
        objects={"sat": SimpleNamespace(enabled=True, kind="satellite")},
    )


def test_cli_output_guard_refuses_nonempty_directory(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    prior = output / "prior.txt"
    prior.write_text("durable evidence", encoding="utf-8")

    with pytest.raises(SystemExit, match="refusing to overwrite or mix prior evidence"):
        run_simulation._prepare_cli_output_directory(_cfg(output), overwrite=False, allow_unsafe=False)

    assert prior.read_text(encoding="utf-8") == "durable evidence"


def test_cli_output_guard_archives_prior_evidence_when_explicit(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "prior.txt").write_text("durable evidence", encoding="utf-8")

    run_simulation._prepare_cli_output_directory(_cfg(output), overwrite=True, allow_unsafe=False)

    archives = list(tmp_path.glob("run.previous-*"))
    assert len(archives) == 1
    assert (archives[0] / "prior.txt").read_text(encoding="utf-8") == "durable evidence"
    assert output.is_dir()
    assert (output / "output_replacement_manifest.json").is_file()


def test_cli_output_guard_refuses_low_disk_headroom(tmp_path, monkeypatch) -> None:
    usage = SimpleNamespace(total=1024, used=1023, free=1)
    monkeypatch.setattr(run_simulation.shutil, "disk_usage", lambda _path: usage)

    with pytest.raises(SystemExit, match="insufficient filesystem headroom"):
        run_simulation._prepare_cli_output_directory(_cfg(tmp_path / "run"), overwrite=False, allow_unsafe=False)


def test_safe_validation_labels_structural_scope(capsys) -> None:
    assert run_simulation._print_config_validation_report(
        "configs/automation_smoke.yaml", import_plugins=False
    )

    output = capsys.readouterr().out
    assert "STRUCTURALLY PARSED" in output
    assert "SKIPPED (safe validation does not import plugins)" in output
    assert "STRUCTURALLY OK" in output


def test_serial_benchmark_uses_private_scratch_outputs(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario_name": "benchmark_scratch",
                "simulator": {"duration_s": 1.0, "dt_s": 1.0},
                "outputs": {"output_dir": str(tmp_path / "user-output"), "review": {"enabled": True}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_run(config_path: str) -> dict:
        cfg = run_simulation.load_simulation_yaml(config_path)
        seen["output_dir"] = cfg.outputs.output_dir
        seen["review_enabled"] = cfg.outputs.review.enabled
        return {"monte_carlo": {"iterations": 1}, "aggregate_stats": {"pass_rate": 1.0, "fail_rate": 0.0}}

    monkeypatch.setattr(run_simulation, "run_simulation_config_file", fake_run)
    monkeypatch.setattr(run_simulation, "_maxrss_bytes", lambda: 0)
    monkeypatch.setattr(run_simulation, "_cpu_time_seconds_including_children", lambda: 0.0)
    monkeypatch.setattr(run_simulation, "_available_memory_bytes", lambda: None)
    monkeypatch.setattr(run_simulation, "_physical_cpu_count", lambda: None)

    run_simulation._print_serial_benchmark(str(config_path), benchmark_runs=1)

    benchmark_output = Path(str(seen["output_dir"]))
    assert benchmark_output != tmp_path / "user-output"
    assert str(benchmark_output.parent.name).startswith("oel-serial-benchmark-")
    assert seen["review_enabled"] is False
    assert not benchmark_output.parent.exists()
    assert "SERIAL MONTE CARLO BENCHMARK" in capsys.readouterr().out
