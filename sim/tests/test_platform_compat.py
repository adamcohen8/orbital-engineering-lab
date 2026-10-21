from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import run_simulation
from sim.core.models import StateTruth
from sim.execution.object_workers import (
    ObjectStepMessage,
    _persistent_object_worker_loop,
)
from sim.platform_compat import (
    max_rss_bytes,
    open_folder,
    process_cpu_time_seconds,
)


def test_windows_folder_opener_preserves_path_with_spaces(tmp_path: Path) -> None:
    folder = tmp_path / "Orbital Engineering Lab" / "run output"
    folder.mkdir(parents=True)
    startfile = Mock()
    popen = Mock()

    opened = open_folder(
        folder,
        platform_name="win32",
        popen=popen,
        startfile=startfile,
    )

    assert opened == folder
    startfile.assert_called_once_with(str(folder))
    popen.assert_not_called()


@pytest.mark.parametrize(
    ("platform_name", "executable"),
    [("darwin", "open"), ("linux", "xdg-open")],
)
def test_posix_folder_openers_use_argument_lists(
    tmp_path: Path,
    platform_name: str,
    executable: str,
) -> None:
    folder = tmp_path / "output with spaces"
    folder.mkdir()
    popen = Mock()

    open_folder(folder, platform_name=platform_name, popen=popen)

    popen.assert_called_once_with([executable, str(folder)])


def test_cli_output_folder_delegates_to_platform_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    folder = tmp_path / "Orbital Engineering Lab" / "Windows output"
    folder.mkdir(parents=True)
    calls: list[Path] = []
    monkeypatch.setattr(run_simulation, "open_folder", lambda path: calls.append(Path(path)))

    assert run_simulation._open_output_folder(folder)
    assert calls == [folder]


def test_resource_metrics_fall_back_without_unix_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sim.platform_compat as compatibility

    monkeypatch.setattr(compatibility, "_resource", None)

    assert max_rss_bytes() is None
    assert process_cpu_time_seconds() >= 0.0


def test_object_worker_payload_is_spawn_serializable() -> None:
    truth = StateTruth(
        position_eci_km=np.array([7000.0, 0.0, 0.0]),
        velocity_eci_km_s=np.array([0.0, 7.5, 0.0]),
        attitude_quat_bn=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body_rad_s=np.zeros(3),
        mass_kg=100.0,
        t_s=0.0,
    )
    message = ObjectStepMessage(
        object_id="target",
        knowledge_base=None,
        initial_truth=truth,
        world_truth_decision={"target": truth},
        t_s=0.0,
        t_next=1.0,
        sample_index=1,
    )

    restored = pickle.loads(pickle.dumps(message))

    assert restored.object_id == "target"
    assert np.array_equal(restored.initial_truth.position_eci_km, truth.position_eci_km)
    assert _persistent_object_worker_loop.__module__ == "sim.execution.object_workers"




def test_pygame_dummy_display_trainer_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pytest.importorskip("pygame")
    from sim.game.dashboard import PygameRPODashboard

    dashboard = PygameRPODashboard(fullscreen=False, title="OEL dummy-display smoke")
    try:
        assert dashboard.screen.get_size() == (1280, 720)
        assert dashboard.closed is False
    finally:
        dashboard.close()
