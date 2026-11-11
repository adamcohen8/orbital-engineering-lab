from __future__ import annotations

import ctypes
import json
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sim.installation import cli, windows_icon
from sim.installation.desktop import ICON_ASSETS, publish_trainer
from sim.installation.paths import InstallationPaths


def installed(tmp_path, profile="game"):
    paths = InstallationPaths(tmp_path / "OEL data 'quoted'", tmp_path / "config")
    paths.ensure()
    paths.current_state.write_text(json.dumps({"current": "0.30.0"}))
    root = paths.version_root("0.30.0")
    root.mkdir(parents=True)
    (root / "installation-record.json").write_text(
        json.dumps({"profile": profile, "runtime": {"created": False, "python": sys.executable}})
    )
    return paths


@pytest.mark.skipif(os.name == "nt", reason="POSIX application fixture")
def test_macos_application_launches_stable_cli_after_update(tmp_path):
    paths = installed(tmp_path)
    receipt = publish_trainer(paths, home=tmp_path, system="darwin")
    app = Path(receipt["path"])
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    icon = app / "Contents/Resources" / info["CFBundleIconFile"]
    assert icon.read_bytes() == (ICON_ASSETS / "trainer-icon.icns").read_bytes()
    executable = app / "Contents/MacOS" / info["CFBundleExecutable"]
    output = tmp_path / "invocation"
    for version in ["0.30.0", "0.31.0"]:
        launcher = paths.launcher / "oel"
        launcher.write_text(f'#!/bin/sh\nprintf "%s %s" "{version}" "$1" > "{output}"\n')
        launcher.chmod(0o755)
        subprocess.run([str(executable)], cwd="/", check=True)
        assert output.read_text() == f"{version} trainer"
    assert publish_trainer(paths, home=tmp_path, system="darwin") == receipt
    assert icon.read_bytes() == (ICON_ASSETS / "trainer-icon.icns").read_bytes()


def test_desktop_preserves_unrelated_app(tmp_path):
    paths = installed(tmp_path)
    app = tmp_path / "Applications/RPO Trainer.app"
    app.mkdir(parents=True)
    (app / "user-file").write_text("preserve")
    with pytest.raises(RuntimeError, match="unrelated"):
        publish_trainer(paths, home=tmp_path, system="darwin")
    assert (app / "user-file").read_text() == "preserve"


def test_linux_entry_and_core_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "menu"))
    paths = installed(tmp_path)
    target = Path(publish_trainer(paths, home=tmp_path, system="linux")["path"])
    assert target == tmp_path / "menu/applications/oel-rpo-trainer.desktop"
    assert "Terminal=false" in target.read_text()
    assert f'Exec="{paths.launcher}/oel" trainer' in target.read_text()
    icon = paths.launcher / "trainer-icon.png"
    assert f"Icon={icon}\n" in target.read_text()
    assert icon.read_bytes() == (ICON_ASSETS / "trainer-icon.png").read_bytes()
    # Desktop artwork remains available after old immutable engine versions retire.
    shutil.rmtree(paths.version_root("0.30.0"))
    assert icon.is_file()
    paths.version_root("0.30.0").mkdir()
    paths.version_root("0.30.0").joinpath("installation-record.json").write_text(json.dumps({"profile": "core"}))
    assert publish_trainer(paths, home=tmp_path, system="linux")["status"] == "skipped"


def test_linux_icon_path_escapes_backslashes_and_refreshes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "menu"))
    paths = InstallationPaths(tmp_path / "data\\quoted", tmp_path / "config")
    paths.ensure()
    paths.current_state.write_text(json.dumps({"current": "0.30.0"}))
    root = paths.version_root("0.30.0")
    root.mkdir()
    (root / "installation-record.json").write_text(json.dumps({"profile": "game"}))
    entry = Path(publish_trainer(paths, home=tmp_path, system="linux")["path"])
    icon = paths.launcher / "trainer-icon.png"
    assert "Icon=" + str(icon).replace("\\", "\\\\") + "\n" in entry.read_text()
    icon.write_bytes(b"outdated")
    publish_trainer(paths, home=tmp_path, system="linux")
    assert icon.read_bytes() == (ICON_ASSETS / "trainer-icon.png").read_bytes()


@pytest.mark.parametrize("system", ["darwin", "win32", "linux"])
def test_core_profile_does_not_create_trainer_icons(tmp_path, system, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "menu"))
    paths = installed(tmp_path, profile="core")
    assert publish_trainer(paths, home=tmp_path, system=system)["status"] == "skipped"
    assert list(paths.launcher.iterdir()) == []
    assert not (tmp_path / "Applications").exists()


def test_windows_publication_embeds_icon_before_stable_launcher_payload(tmp_path, monkeypatch):
    import pip._vendor.distlib as distlib
    import pip._vendor.distlib.scripts as scripts

    paths = installed(tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    stock = Path(distlib.__file__).with_name("w64.exe").read_bytes()
    icon_calls = []

    class WindowsScriptMaker(scripts.ScriptMaker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._is_nt = True

        def _build_shebang(self, executable, post_interp):
            # Windows distlib always emits a direct interpreter shebang.
            # Only stub the platform branch; still exercise extraction of flags.
            return b"#!" + executable + post_interp + b"\n"

        def _get_launcher(self, kind):
            assert kind == "w"
            return stock

    def embed(launcher, icon, work_dir):
        icon_calls.append(icon)
        assert launcher == stock
        assert icon.read_bytes() == (ICON_ASSETS / "trainer-icon.ico").read_bytes()
        return launcher + b"ICON-BEFORE-ZIP"

    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "-c":
            exec(compile(argv[2], "windows-launcher-generator", "exec"), {})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scripts, "ScriptMaker", WindowsScriptMaker)
    monkeypatch.setattr(windows_icon, "launcher_with_icon", embed)
    monkeypatch.setattr(subprocess, "run", run)
    executable = Path(publish_trainer(paths, home=tmp_path, system="win32")["path"])
    payload = executable.read_bytes()
    launcher_prefix = stock + b"ICON-BEFORE-ZIP"
    assert payload.startswith(launcher_prefix)
    # distlib puts the interpreter shebang between the native launcher and its
    # zip payload.  Keep the GUI bootstrap isolated before it reaches the
    # stable launcher, which performs the managed-tree authentication.
    shebang = payload[len(launcher_prefix) :].splitlines()[0]
    assert shebang.startswith(b"#!")
    assert b"-I" in shebang.split(maxsplit=1)[-1]
    assert b"-B" in shebang.split(maxsplit=1)[-1]
    with zipfile.ZipFile(executable) as archive:
        script = archive.read("__main__.py")
    stable_launcher = paths.launcher / ("oel.cmd" if os.name == "nt" else "oel")
    assert str(stable_launcher).encode() in script
    assert b'subprocess.call([str(launcher), "trainer"]' in script
    assert b"state/current.json" not in script
    icon = paths.launcher / "trainer-icon.ico"
    assert icon_calls == [icon]
    shortcut_command, shortcut_options = calls[-1]
    assert shortcut_options["env"]["OEL_TRAINER_ICON"] == str(icon)
    assert '$s.IconLocation="$env:OEL_TRAINER_ICON,0"' in shortcut_command[-1]


def test_packaged_icon_formats_share_approved_master():
    from PIL import Image

    with Image.open(ICON_ASSETS / "trainer-icon.png") as master:
        assert master.size == (1024, 1024)
        with Image.open(ICON_ASSETS / "trainer-icon.ico") as windows:
            assert windows.ico.sizes() == {(size, size) for size in [16, 24, 32, 48, 64, 128, 256]}
            assert windows.convert("RGBA").tobytes() == master.resize((256, 256), Image.Resampling.LANCZOS).convert("RGBA").tobytes()
        with Image.open(ICON_ASSETS / "trainer-icon.icns") as mac:
            assert mac.size == (1024, 1024)
            assert mac.convert("RGBA").tobytes() == master.convert("RGBA").tobytes()


def test_windows_icon_group_references_every_ico_image():
    icon = (ICON_ASSETS / "trainer-icon.ico").read_bytes()
    resources = windows_icon.icon_resources(icon)
    kind, group_id, group = resources[-1]
    assert (kind, group_id) == (14, 101)
    assert struct.unpack_from("<HHH", group) == (0, 1, 7)
    for index, (kind, identifier, payload) in enumerate(resources[:-1]):
        assert kind == 3
        group_entry = group[6 + index * 14 : 20 + index * 14]
        size, reference = struct.unpack_from("<IH", group_entry, 8)
        assert reference == identifier == index + 1
        assert size == len(payload)
        assert group_entry[:8] == icon[6 + index * 16 : 14 + index * 16]


@pytest.mark.parametrize("icon", [b"", b"\0" * 6, b"\0\0\1\0\1\0", b"\0\0\1\0\1\0" + b"\0" * 16])
def test_windows_rejects_invalid_icon_before_resource_update(icon):
    with pytest.raises(ValueError):
        windows_icon.icon_resources(icon)


@pytest.mark.parametrize("fail_update", [False, True])
def test_windows_resource_update_commits_or_discards(tmp_path, monkeypatch, fail_update):
    from unittest.mock import Mock

    calls = []
    kernel = SimpleNamespace(
        BeginUpdateResourceW=Mock(return_value=123),
        UpdateResourceW=Mock(),
        EndUpdateResourceW=Mock(return_value=True),
    )

    def update(handle, kind, identifier, language, data, size):
        calls.append((kind, identifier, ctypes.string_at(data, size)))
        return not fail_update

    kernel.UpdateResourceW.side_effect = update
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError("resource update failed"), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    icon = ICON_ASSETS / "trainer-icon.ico"
    if fail_update:
        with pytest.raises(OSError, match="resource update failed"):
            windows_icon.launcher_with_icon(b"MZfixture", icon, tmp_path)
    else:
        assert windows_icon.launcher_with_icon(b"MZfixture", icon, tmp_path) == b"MZfixture"
        assert calls == windows_icon.icon_resources(icon.read_bytes())
    kernel.EndUpdateResourceW.assert_called_once_with(123, fail_update)


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows PE resource and GUI launcher smoke")
def test_native_windows_icon_launcher_preserves_executable_payload(tmp_path):
    from ctypes import wintypes

    from pip._vendor.distlib.scripts import ScriptMaker

    script = tmp_path / "trainer-smoke.py"
    output = tmp_path / "launched.txt"
    script.write_text(f"from pathlib import Path\nPath({str(output)!r}).write_text('launched')\n")
    maker = ScriptMaker("", str(tmp_path))
    maker.executable = sys.executable
    original = maker._get_launcher
    maker._get_launcher = lambda kind: windows_icon.launcher_with_icon(
        original(kind), ICON_ASSETS / "trainer-icon.ico", tmp_path
    )
    maker.make(str(script), options={"gui": True})
    executable = tmp_path / "trainer-smoke.exe"
    with zipfile.ZipFile(executable) as archive:
        assert archive.read("__main__.py") == script.read_bytes()
    subprocess.run([str(executable)], check=True, timeout=30)
    assert output.read_text() == "launched"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel.LoadLibraryExW.restype = wintypes.HMODULE
    kernel.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.FindResourceW.restype = wintypes.HANDLE
    kernel.SizeofResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.SizeofResource.restype = wintypes.DWORD
    kernel.LoadResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.LoadResource.restype = wintypes.HANDLE
    kernel.LockResource.argtypes = [wintypes.HANDLE]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    handle = kernel.LoadLibraryExW(str(executable), None, 2)
    assert handle
    try:
        for kind, identifier, expected in windows_icon.icon_resources((ICON_ASSETS / "trainer-icon.ico").read_bytes()):
            resource = kernel.FindResourceW(handle, identifier, kind)
            assert resource
            size = kernel.SizeofResource(handle, resource)
            data = kernel.LockResource(kernel.LoadResource(handle, resource))
            assert ctypes.string_at(data, size) == expected
    finally:
        kernel.FreeLibrary(handle)


@pytest.mark.skipif(os.name == "nt", reason="POSIX fake interpreter fixture")
def test_windows_desktop_script_forwards_to_activation_bound_launcher_each_launch(tmp_path):
    paths = installed(tmp_path)
    # The desktop script delegates selection and preflight to the stable
    # launcher.  Each fake launcher below represents the version-bound
    # launcher rewritten by activation/update.
    publish_trainer(paths, home=tmp_path, system="darwin")
    script = paths.launcher / "rpo-trainer.py"
    launcher = paths.launcher / "oel"
    output = tmp_path / "invocation"
    for version in ["0.30.0", "0.31.0"]:
        paths.current_state.write_text(json.dumps({"current": version}))
        launcher.write_text(
            "#!/bin/sh\n"
            f'printf "%s %s" "{version}" "$1" > "{output}"\n'
        )
        launcher.chmod(0o755)
        subprocess.run([sys.executable, str(script)], check=True)
        assert output.read_text() == f"{version} trainer"


def test_trainer_dispatch_uses_managed_engine_and_writable_cwd(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    source = tmp_path / "source"
    monkeypatch.setattr(cli, "_selected_engine", lambda *args: ("0.30.0", source, Path(sys.executable), "developer"))
    calls = []
    monkeypatch.setattr(
        cli.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0)
    )
    assert cli._dispatch("trainer", ["--help"], paths=paths, workspace_path=None) == 0
    assert calls[0][0] == [sys.executable, str(source / "run_game.py"), "--help"]
    assert calls[0][1]["cwd"] == paths.data_root
    assert calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable fixture")
def test_trainer_ignores_incidental_workspace(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    monkeypatch.setattr(cli, "_find_workspace", lambda: tmp_path / "incidental/oel-workspace.yaml")
    seen = []
    monkeypatch.setattr(cli, "_dispatch", lambda command, arguments, **kwargs: seen.append(kwargs) or 0)
    assert cli.main(["--data-root", str(paths.data_root), "--config-root", str(paths.config_root), "trainer"]) == 0
    assert seen[0]["workspace_path"] is None


def test_windows_launcher_is_a_gui_executable_forwarding_to_stable_launcher(tmp_path):
    import zipfile

    import pip._vendor.distlib as distlib
    from pip._vendor.distlib.scripts import ScriptMaker

    paths = installed(tmp_path)
    publish_trainer(paths, home=tmp_path, system="darwin")
    destination = tmp_path / "windows"
    destination.mkdir()
    maker = ScriptMaker("", str(destination))
    maker.executable = r"C:\Python\python.exe"
    maker._is_nt = True
    launcher_kind = []

    def launcher(kind):
        launcher_kind.append(kind)
        return Path(distlib.__file__).with_name("w64.exe").read_bytes()

    maker._get_launcher = launcher
    maker.make(str(paths.launcher / "rpo-trainer.py"), options={"gui": True})
    executable = destination / "rpo-trainer.exe"
    assert executable.read_bytes().startswith(b"MZ")
    assert launcher_kind == ["w"]
    with zipfile.ZipFile(executable) as archive:
        script = archive.read("__main__.py")
    compile(script, "__main__.py", "exec")
    stable_launcher = paths.launcher / ("oel.cmd" if os.name == "nt" else "oel")
    assert str(stable_launcher).encode() in script
    assert b'subprocess.call([str(launcher), "trainer"]' in script
    assert b"state/current.json" not in script


def test_update_defaults_to_existing_game_profile(tmp_path, monkeypatch):
    paths = installed(tmp_path)
    seen = []
    monkeypatch.setattr(cli, "install_release", lambda *args, **kw: seen.append(kw) or {})
    monkeypatch.setattr(cli, "_find_workspace", lambda: None)
    assert (
        cli.main(
            [
                "--data-root",
                str(paths.data_root),
                "--config-root",
                str(paths.config_root),
                "update",
                "install",
                "release.json",
                "--developer-unsigned",
                "--no-runtime",
            ]
        )
        == 0
    )
    assert seen[0]["profile"] == "game"
