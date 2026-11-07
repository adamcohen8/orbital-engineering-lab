from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import sim.agent_task.__main__ as agent_task_cli
import sim.agent_task.runner as agent_task_runner
from sim.agent_task.models import AgentTaskRecipe
from sim.execution.run_lifecycle.models import RunPolicyError
from sim.execution.run_lifecycle.runner import _artifact_inventory, prepare_foreground_run
from sim.execution.run_lifecycle.store import LifecycleStore
from sim.installation.contracts import ContractError
from sim.installation.workspace import init_workspace, load_workspace
from sim.security.output_writes import SecureWriteError

ROOT = Path(__file__).resolve().parents[2]


def test_agent_task_inspect_cli_preserves_omitted_query_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    def fake_inspect(output_dir: str, *, query_names: object, max_rows: int) -> dict[str, object]:
        calls.append(query_names)
        return {"status": "completed"}

    monkeypatch.setattr(agent_task_cli, "inspect_output", fake_inspect)
    monkeypatch.setattr(agent_task_cli, "_print_packet", lambda *_args, **_kwargs: None)

    assert agent_task_cli.main(["inspect", str(tmp_path), "--json"]) == 0
    assert calls == [None]

    assert agent_task_cli.main(["inspect", str(tmp_path), "--query", "run_metadata", "--json"]) == 0
    assert calls[-1] == ("run_metadata",)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable on this platform")
@pytest.mark.parametrize("dangling", [False, True])
def test_workspace_manifest_rejects_symlink_path_components(tmp_path: Path, dangling: bool) -> None:
    workspace = tmp_path / ("workspace-dangling" if dangling else "workspace-in-root")
    init_workspace(workspace, engine_version="0.36.0")
    target = workspace / "configs" / "real"
    target.mkdir()
    link = workspace / "configs" / "alias"
    link.symlink_to(workspace / "configs" / "missing" if dangling else target, target_is_directory=True)

    manifest = yaml_load(workspace / "oel-workspace.yaml")
    manifest["paths"]["configs"] = "configs/alias"
    (workspace / "oel-workspace.yaml").write_text(
        yaml_dump(manifest), encoding="utf-8"
    )

    with pytest.raises(ContractError, match="symbolic link"):
        load_workspace(workspace)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable on this platform")
def test_lifecycle_preparation_rejects_in_root_symlink_before_resolution(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    real = output_root / "real"
    real.mkdir(parents=True)
    alias = output_root / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(RunPolicyError, match="symlink"):
        prepare_foreground_run(
            config_path=ROOT / "configs" / "automation_smoke.yaml",
            output_dir=alias / "run",
            output_root=output_root,
            workspace_root=ROOT,
        )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable on this platform")
def test_lifecycle_store_rejects_dangling_symlink_before_creation(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    dangling = output_root / "dangling"
    target = output_root / "missing"
    dangling.symlink_to(target, target_is_directory=True)
    store = LifecycleStore(state_root=tmp_path / "state", allowed_output_roots=(output_root,))

    with pytest.raises(RunPolicyError, match="symlink"):
        store.prepare_output_dir(dangling / "run")
    assert not target.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable on this platform")
@pytest.mark.parametrize("dangling", [False, True])
def test_lifecycle_artifact_inventory_never_hashes_symlinks(tmp_path: Path, dangling: bool) -> None:
    output = tmp_path / "output"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_text('{"private": true}\n', encoding="utf-8")
    link = output / ("dangling.json" if dangling else "external.json")
    link.symlink_to(tmp_path / "missing.json" if dangling else external)

    with pytest.raises(SecureWriteError, match="regular files and directories"):
        _artifact_inventory(output)


def test_generic_recipe_final_cancellation_persists_packet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "recipe"
    recipe = AgentTaskRecipe(
        recipe_id="fake_recipe",
        title="Fake recipe",
        description="focused cancellation regression",
        config_path="fake.yaml",
    )
    prepared = {
        "output_dir": str(output),
        "config_path": str(output / "prepared.yaml"),
        "source_config": str(output / "source.yaml"),
    }

    class Workspace:
        def validate(self, _path: str) -> dict[str, object]:
            return {"ok": True}

    monkeypatch.setattr(agent_task_runner, "get_recipe", lambda _recipe_id: recipe)
    monkeypatch.setattr(agent_task_runner, "prepare_recipe_config", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(agent_task_runner, "_workspace", lambda: Workspace())
    monkeypatch.setattr(agent_task_runner, "_run_prepared_config", lambda *_args, **_kwargs: {"scenario_name": "fake"})
    monkeypatch.setattr(
        agent_task_runner,
        "inspect_output",
        lambda *_args, **_kwargs: {"status": "completed", "review": {}, "artifacts": [], "artifact_summary": {}},
    )

    def cancel_on_final(done: int, total: int) -> None:
        if (done, total) == (1, 1):
            raise agent_task_runner.AgentTaskCancelled("final callback")

    packet = agent_task_runner.run_recipe("fake_recipe", output_dir=output, step_callback=cancel_on_final)

    assert packet["status"] == "cancelled"
    saved = json.loads((output / "agent_evidence_packet.json").read_text(encoding="utf-8"))
    assert saved["status"] == "cancelled"
    assert "partial" in " ".join(saved["caveats"])


def yaml_load(path: Path) -> dict[str, object]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def yaml_dump(value: dict[str, object]) -> str:
    import yaml

    return yaml.safe_dump(value, sort_keys=False)
