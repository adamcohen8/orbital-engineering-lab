"""Per-user RPO Trainer desktop integration for verified managed installs."""

from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from sim.platform_compat import canonicalize_platform_compatibility_root as _canonicalize_platform_root
from sim.security.output_writes import SecureWriteRoot, secure_write_primitives_available

from .paths import InstallationPaths

MARKER = "oel.rpo-trainer.desktop.v1"
ICON_ASSETS = Path(__file__).with_name("data")


def publish_trainer(paths: InstallationPaths, *, home: Path | None = None, system: str | None = None) -> dict[str, str]:
    """Publish a launcher after activation; never replace an unrelated application."""
    home = home or Path.home()
    system = system or sys.platform
    state = json.loads(paths.current_state.read_text())
    record = json.loads((paths.version_root(state["current"]) / "installation-record.json").read_text())
    if record.get("profile") not in {"game", "full", "cross-platform"}:
        return {"status": "skipped", "reason": "Trainer requires the game dependency profile"}
    # Route every desktop surface through the stable launcher.  The launcher
    # performs external signed source/runtime preflight before it imports the
    # selected managed CLI; this script must never execute a managed runtime
    # directly from a mutable installation record.
    script = paths.launcher / "rpo-trainer.py"
    stable_launcher = paths.launcher / ("oel.cmd" if os.name == "nt" else "oel")
    script_text = (
        # distlib copies filename shebang arguments into the embedded Windows
        # launcher; its filename path does not receive ``make()`` options.
        "#!/usr/bin/env pythonw -I -B\nimport os, pathlib, subprocess\n"
        f"launcher = pathlib.Path({str(stable_launcher)!r})\n"
        f"data = pathlib.Path({str(paths.data_root)!r})\n"
        'logs = data / "logs"\nlogs.mkdir(parents=True, exist_ok=True)\n'
        'with (logs / "trainer.log").open("a") as log:\n'
        '    raise SystemExit(subprocess.call([str(launcher), "trainer"], '
        "cwd=data, stdout=log, stderr=log, "
        'creationflags=0x08000000 if os.name == "nt" else 0))\n'
    )
    # Validate and retain the managed launcher directory before any later
    # path-facing ``ensure`` calls.  A pre-existing launcher symlink must not
    # redirect this first file write outside the data root.
    _write_launcher_bytes(paths, script.name, script_text.encode("utf-8"))
    paths.ensure()
    if system == "darwin":
        target = home / "Applications" / "RPO Trainer.app"
        marker = target / "Contents" / "oel-owner.json"
        _reject_symlink_ancestors(target)
        _check_owner(target, marker, paths)
        contents = target / "Contents"
        (contents / "MacOS").mkdir(parents=True, exist_ok=True)
        resources = contents / "Resources"
        resources.mkdir(exist_ok=True)
        shutil.copyfile(ICON_ASSETS / "trainer-icon.icns", resources / "trainer-icon.icns")
        executable = contents / "MacOS" / "RPO Trainer"
        log_dir = paths.data_root / "logs"
        executable.write_text(
            "#!/bin/sh\nmkdir -p "
            + shlex.quote(str(log_dir))
            + "\nexec "
            + shlex.quote(str(paths.launcher / "oel"))
            + " trainer >> "
            + shlex.quote(str(log_dir / "trainer.log"))
            + " 2>&1\n"
        )
        executable.chmod(0o755)
        (contents / "Info.plist").write_bytes(
            plistlib.dumps(
                {
                    # Keep the installed app identity stable across the product rename.
                    "CFBundleIdentifier": "org.orbitalengagementlab.trainer",
                    "CFBundleName": "RPO Trainer",
                    "CFBundleExecutable": "RPO Trainer",
                    "CFBundlePackageType": "APPL",
                    "CFBundleVersion": "1",
                    "CFBundleIconFile": "trainer-icon.icns",
                    "NSHighResolutionCapable": True,
                }
            )
        )
    elif system == "win32":
        target = paths.launcher / "RPO Trainer.exe"
        marker = paths.launcher / "rpo-trainer-owner.json"
        _reject_symlink_ancestors(target)
        _check_owner(target, marker, paths)
        start_menu = Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Microsoft/Windows/Start Menu/Programs"
        start_menu.mkdir(parents=True, exist_ok=True)
        shortcut = start_menu / "RPO Trainer.lnk"
        _reject_symlink_ancestors(shortcut)
        if shortcut.exists() and not marker.exists():
            raise RuntimeError(f"Refusing to replace an unrelated shortcut: {shortcut}")
        # The operator-owned provider's pip distlib generates the
        # platform/architecture-specific GUI executable without a compiler.
        # Script generation itself must not execute the selected managed
        # runtime.  Use the operator-owned provider that also runs the stable
        # preflight dispatcher; the generated GUI executable invokes the
        # stable launcher script when the user opens Trainer.
        base_python = Path(getattr(sys, "_base_executable", sys.executable)).expanduser().absolute()
        try:
            base_python.resolve(strict=False).relative_to(paths.versions.resolve(strict=False))
        except ValueError:
            pass
        else:
            raise RuntimeError("Desktop launcher requires an external Python provider outside managed versions")
        if not base_python.is_file():
            raise RuntimeError(f"Desktop launcher Python provider is not a regular file: {base_python}")
        windowed_python = base_python.with_name("pythonw.exe")
        if windowed_python.is_file():
            base_python = windowed_python
        icon = paths.launcher / "trainer-icon.ico"
        _write_launcher_bytes(paths, icon.name, (ICON_ASSETS / "trainer-icon.ico").read_bytes())
        with tempfile.TemporaryDirectory(dir=paths.cache) as temp:
            generator = (
                "from pathlib import Path\n"
                "from pip._vendor.distlib.scripts import ScriptMaker, enquote_executable\n"
                "from sim.installation.windows_icon import launcher_with_icon\n"
                f'm=ScriptMaker("", {temp!r})\nm.executable=enquote_executable({str(base_python)!r})\n'
                "original=m._get_launcher\n"
                f"m._get_launcher=lambda kind: launcher_with_icon(original(kind), Path({str(icon)!r}), Path({temp!r}))\n"
                # The source filename carries the isolation flags because
                # distlib's filename-copy path derives the shebang from it.
                f'm.make({str(script)!r}, options={{"gui": True}})\n'
            )
            subprocess.run([str(base_python), "-c", generator], check=True)
            _write_launcher_bytes(paths, target.name, (Path(temp) / "rpo-trainer.exe").read_bytes())
        env = dict(os.environ, OEL_TRAINER_LINK=str(shortcut), OEL_TRAINER_EXE=str(target), OEL_TRAINER_ICON=str(icon))
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:OEL_TRAINER_LINK); "
                'if ((Test-Path $env:OEL_TRAINER_LINK) -and $s.TargetPath -ne $env:OEL_TRAINER_EXE) { throw "Unrelated shortcut" }; '
                '$s.TargetPath=$env:OEL_TRAINER_EXE; $s.IconLocation="$env:OEL_TRAINER_ICON,0"; '
                '$s.Description="OEL RPO Trainer"; $s.Save()',
            ],
            env=env,
            check=True,
        )
    else:
        target = (
            Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "applications" / "oel-rpo-trainer.desktop"
        )
        marker = target.with_suffix(".owner.json")
        _reject_symlink_ancestors(target)
        _check_owner(target, marker, paths)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Desktop Exec fields use their own quoting rules, not shell quoting.
        command = (
            str(paths.launcher / "oel")
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("`", "\\`")
            .replace("$", "\\$")
            .replace("%", "%%")
        )
        if any(c in str(paths.launcher) for c in ("\n", "\r", "\t", "=")):
            raise ValueError("Desktop launcher path contains unsupported characters")
        icon = paths.launcher / "trainer-icon.png"
        _write_launcher_bytes(paths, icon.name, (ICON_ASSETS / "trainer-icon.png").read_bytes())
        icon_value = str(icon).replace("\\", "\\\\")
        target.write_text(
            "[Desktop Entry]\nType=Application\nName=RPO Trainer\n"
            "Comment=Orbital Engineering Lab RPO Trainer\n"
            f'Exec="{command}" trainer\nIcon={icon_value}\nTerminal=false\nCategories=Education;Science;\n'
        )
        target.chmod(0o755)
    marker_payload = json.dumps({"owner": MARKER, "data_root": str(paths.data_root)}).encode("utf-8")
    if system == "win32":
        _write_launcher_bytes(paths, marker.name, marker_payload)
    else:
        marker.write_bytes(marker_payload)
    if system == "darwin":
        # Finder refreshes cached bundle icons when the application changes.
        target.touch()
    return {"status": "ready", "path": str(target)}


def _check_owner(target: Path, marker: Path, paths: InstallationPaths) -> None:
    if _is_linked(target) or _is_linked(marker):
        raise RuntimeError(f"Refusing linked desktop launcher: {target}")
    if target.exists():
        expected = {"owner": MARKER, "data_root": str(paths.data_root)}
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise RuntimeError(f"Refusing to replace an unrelated desktop launcher: {target}")


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject desktop destinations routed through a pre-existing symlink.

    Checking only the launcher and marker leaves a linked ``Applications`` or
    ``XDG_DATA_HOME`` parent able to redirect installation writes outside the
    user-selected destination.  Walk the lexical parent chain before any
    directory creation; nonexistent parents are safe and will be created by
    the caller after this check.
    """

    current = _canonicalize_platform_root(path).parent
    while True:
        if _is_linked(current):
            raise RuntimeError(f"Refusing linked desktop launcher parent: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _is_linked(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows junction/reparse point."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            return bool(is_junction())
        except OSError:
            return True
    return False


def _write_launcher_bytes(paths: InstallationPaths, name: str, payload: bytes) -> Path:
    """Write one managed launcher file without following its parent."""

    target = paths.launcher / name
    if not secure_write_primitives_available():
        # Windows and other hosts retain the compatibility path.  Static
        # symlink/junction checks reject pre-existing redirection, but there
        # is no portable hostile-race guarantee without a platform descriptor.
        _reject_symlink_ancestors(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target
    launcher_root = _canonicalize_platform_root(paths.launcher)
    with SecureWriteRoot.open(
        launcher_root,
        approved_roots=(Path(os.path.sep),),
        create=True,
    ) as root:
        try:
            metadata = os.stat(name, dir_fd=root.fd, follow_symlinks=False)
        except FileNotFoundError:
            root.write_bytes(name, payload)
        else:
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"Refusing linked or non-regular desktop launcher file: {target}")
            root.adopt_existing(name)
            root.replace_owned_bytes(name, payload)
    return target
