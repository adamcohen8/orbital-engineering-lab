"""Bootstrap-safe environment diagnostics for the OEL command-line interface.

Keep this module limited to the Python standard library and lightweight OEL
metadata helpers. The doctor must remain usable when scientific dependencies
are missing or incompatible.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from sim.installation.resources import ResourceNotFoundError, quickstart_config_path
from sim.project_version import inspect_project_version

SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 15)
SUPPORTED_PYTHON_RANGE = ">=3.10,<3.15"

# CPython publishes security-fix windows by feature release. Keep this table
# explicit so the functional range can outlive an upstream security baseline
# without silently changing OEL's packaging contract.
PYTHON_SECURITY_EOL_MONTH = {
    (3, 10): (2026, 10),
    (3, 11): (2027, 10),
    (3, 12): (2028, 10),
    (3, 13): (2029, 10),
    (3, 14): (2030, 10),
}

DOCTOR_CONTRACT_LABELS = (
    "Functional Python",
    "Security baseline",
    "Operating system",
    "OS version",
    "Architecture",
    "Python executable",
    "OEL version",
    "Installation state",
    "Workspace pin",
    "Core dependencies",
    "Dependency graph",
    "Install profile",
    "Core runtime",
    "Development/test",
    "Trainer",
    "Acceleration",
    "OGP validation",
    "Machine learning",
    "Quickstart config",
    "Output directory",
    "Quickstart validation",
)


@dataclass(frozen=True)
class DependencySpec:
    distribution: str
    minimum: str | None = None
    maximum_exclusive: str | None = None
    exact: str | None = None

    @property
    def requirement(self) -> str:
        if self.exact is not None:
            return f"=={self.exact}"
        parts: list[str] = []
        if self.minimum is not None:
            parts.append(f">={self.minimum}")
        if self.maximum_exclusive is not None:
            parts.append(f"<{self.maximum_exclusive}")
        return ",".join(parts) or "installed"


@dataclass(frozen=True)
class DependencyStatus:
    spec: DependencySpec
    version: str | None
    compatible: bool

    @property
    def detail(self) -> str:
        if self.version is None:
            return f"missing (requires {self.spec.requirement})"
        if self.compatible:
            return f"{self.version} ({self.spec.requirement})"
        return f"{self.version} is incompatible (requires {self.spec.requirement})"


def _pillow_spec(version: tuple[int, int]) -> DependencySpec:
    return DependencySpec("Pillow", exact="12.3.0") if version < (3, 11) else DependencySpec("Pillow", "13", "14")


CORE_SPECS = (
    DependencySpec("oel-rust-game", "0.5", "0.6"),
    DependencySpec("oel-rust-orbit", "0.18.3", "0.19"),
    DependencySpec("numpy", "2.1", "2.5"),
    DependencySpec("matplotlib", "3.8", "3.12"),
    _pillow_spec(tuple(sys.version_info[:2])),
    DependencySpec("tqdm", "4.65", "5"),
    DependencySpec("PyYAML", "6.0", "7"),
    DependencySpec("scipy", "1.14.1", "1.18"),
    DependencySpec("pyerfa", "2.0.1.5", "3"),
)
DEV_SPECS_BASE = (
    DependencySpec("pytest", "9", "10"),
    DependencySpec("ruff", exact="0.15.14"),
    DependencySpec("setuptools", "83", "84"),
)
GAME_MEDIA_SPECS = (
    _pillow_spec(tuple(sys.version_info[:2])),
    DependencySpec("imageio", "2.34", "3"),
    DependencySpec("imageio-ffmpeg", "0.4", "1"),
)
ACCEL_SPECS = (DependencySpec("numba", "0.61", "0.67"),)
VALIDATION_SPECS = (DependencySpec("sgp4", "2.24", "3"),)
ML_SPECS = (
    DependencySpec("gymnasium", "0.29", "2"),
    DependencySpec("torch", "2.13", "2.14"),
    DependencySpec("filelock", "3.20.3", "4"),
)


def doctor_requested(arguments: Sequence[str]) -> bool:
    return "--doctor" in arguments


def interpreter_is_supported(version_info: Sequence[int] | None = None) -> bool:
    version = tuple(version_info or sys.version_info[:2])[:2]
    return SUPPORTED_PYTHON_MIN <= version < SUPPORTED_PYTHON_MAX_EXCLUSIVE


def security_support_detail(
    version_info: Sequence[int] | None = None,
    *,
    today: date | None = None,
) -> tuple[bool, str]:
    version = tuple(version_info or sys.version_info[:2])[:2]
    eol_month = PYTHON_SECURITY_EOL_MONTH.get(version)
    version_text = ".".join(str(item) for item in version)
    if eol_month is None:
        return False, f"CPython {version_text} is outside OEL's recorded security baseline."
    current = today or date.today()
    end_year, end_month = eol_month
    after_eol = (current.year, current.month) > (end_year, end_month)
    if after_eol:
        return (
            False,
            f"CPython {version_text} is a functional legacy tier; upstream security maintenance ended "
            f"{end_year:04d}-{end_month:02d}.",
        )
    return (
        True,
        f"CPython {version_text} receives upstream security maintenance through "
        f"{end_year:04d}-{end_month:02d}; release dependency-audit evidence is still required.",
    )


def _version_key(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", str(value))
    if match is None:
        return ()
    return tuple(int(item) for item in match.group(1).split("."))


def _compare_version(left: str, right: str) -> int:
    left_key = _version_key(left)
    right_key = _version_key(right)
    width = max(len(left_key), len(right_key), 1)
    left_key += (0,) * (width - len(left_key))
    right_key += (0,) * (width - len(right_key))
    return (left_key > right_key) - (left_key < right_key)


def dependency_is_compatible(version: str, spec: DependencySpec) -> bool:
    if not _version_key(version):
        return False
    if spec.exact is not None and _compare_version(version, spec.exact) != 0:
        return False
    if spec.minimum is not None and _compare_version(version, spec.minimum) < 0:
        return False
    if spec.maximum_exclusive is not None and _compare_version(version, spec.maximum_exclusive) >= 0:
        return False
    return True


def _installed_version(distribution: str) -> str | None:
    try:
        return str(metadata.version(distribution) or "").strip() or None
    except (metadata.PackageNotFoundError, TypeError, KeyError, AttributeError, ValueError):
        return None


def evaluate_dependencies(
    specs: Iterable[DependencySpec],
    *,
    installed_versions: Mapping[str, str | None] | None = None,
) -> tuple[DependencyStatus, ...]:
    supplied = installed_versions or {}
    statuses: list[DependencyStatus] = []
    for spec in specs:
        if installed_versions is None:
            version = _installed_version(spec.distribution)
        else:
            version = supplied.get(spec.distribution)
        statuses.append(
            DependencyStatus(
                spec=spec,
                version=version,
                compatible=version is not None and dependency_is_compatible(version, spec),
            )
        )
    return tuple(statuses)


def _core_specs(version: tuple[int, int]) -> tuple[DependencySpec, ...]:
    return tuple(_pillow_spec(version) if spec.distribution == "Pillow" else spec for spec in CORE_SPECS)


def _dev_specs(version: tuple[int, int]) -> tuple[DependencySpec, ...]:
    if version < (3, 11):
        return DEV_SPECS_BASE + (DependencySpec("tomli", "2", "3"),)
    return DEV_SPECS_BASE


def _game_specs(version: tuple[int, int]) -> tuple[DependencySpec, ...]:
    pygame_spec = (
        DependencySpec("pygame-ce", "2.5", "3")
        if version >= (3, 14)
        else DependencySpec("pygame", "2.5", "3")
    )
    media_specs = (_pillow_spec(version),) + GAME_MEDIA_SPECS[1:]
    return (pygame_spec,) + media_specs


def _all_compatible(statuses: Iterable[DependencyStatus]) -> bool:
    return all(item.compatible for item in statuses)


def _failed_dependency_detail(statuses: Iterable[DependencyStatus]) -> str:
    missing = [item.spec.distribution for item in statuses if item.version is None]
    incompatible = [
        f"{item.spec.distribution} {item.version} (requires {item.spec.requirement})"
        for item in statuses
        if item.version is not None and not item.compatible
    ]
    parts: list[str] = []
    if missing:
        parts.append(
            "missing distributions/wheels: " + ", ".join(missing)
        )
    if incompatible:
        parts.append("incompatible versions: " + ", ".join(incompatible))
    return "; ".join(parts) or "unknown dependency failure"


def _capability_detail(
    statuses: tuple[DependencyStatus, ...],
    *,
    install_extra: str,
    available_detail: str,
) -> tuple[bool, str]:
    if _all_compatible(statuses):
        return True, available_detail
    return (
        False,
        f"{_failed_dependency_detail(statuses)}; install with `.[{install_extra}]` using the matching constraints file.",
    )


def _linux_distribution() -> tuple[str, str]:
    try:
        data = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        data = {}
    return str(data.get("ID", "")).lower(), str(data.get("VERSION_ID", ""))


def _platform_is_supported(system: str, machine: str) -> tuple[bool, str]:
    system_key = system.lower()
    machine_key = machine.lower()
    x64 = {"x86_64", "amd64"}
    if system_key == "windows":
        version_numbers = [int(item) for item in re.findall(r"\d+", platform.version())]
        build = version_numbers[2] if len(version_numbers) >= 3 else 0
        windows_11 = build >= 22000
        windows_server_2022 = build == 20348
        ok = machine_key in x64 and (windows_11 or windows_server_2022)
        expected = "Windows 11 x64 or Windows Server 2022 x64 automation"
    elif system_key == "linux":
        distribution, version = _linux_distribution()
        ok = machine_key in x64 and distribution == "ubuntu" and version in {"22.04", "24.04"}
        expected = "Ubuntu 22.04/24.04 x64"
    elif system_key == "darwin":
        mac_version = _version_key(platform.mac_ver()[0])
        ok = machine_key in x64 | {"arm64", "aarch64"} and bool(mac_version) and mac_version >= (14,)
        expected = "macOS 14+ arm64 or x64"
    else:
        ok = False
        expected = (
            "Windows 11 x64, Windows Server 2022 x64 automation, "
            "Ubuntu 22.04/24.04 x64, or macOS 14+ arm64/x64"
        )
    return ok, expected


def _acceleration_qualified(system: str, machine: str) -> bool:
    return not (system.lower() == "darwin" and machine.lower() in {"x86_64", "amd64"})


def _os_version_detail(system: str) -> str:
    if system.lower() == "darwin":
        mac_version = platform.mac_ver()[0] or "unknown"
        return f"macOS {mac_version}; Darwin {platform.release() or 'unknown'}"
    if system.lower() == "linux":
        distribution, version = _linux_distribution()
        name = distribution or "unknown distribution"
        return f"{name} {version or 'unknown'}; kernel {platform.release() or 'unknown'}"
    if system.lower() == "windows":
        return f"Windows {platform.release() or 'unknown'}; build {platform.version() or 'unknown'}"
    return platform.version() or platform.release() or "unknown"


def _pip_check() -> tuple[bool, str]:
    if _installed_version("pip") is None:
        return False, "pip is unavailable; dependency consistency could not be checked."
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except Exception as exc:
        return False, f"`python -m pip check` could not run: {exc}"
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode == 0:
        return True, output or "No broken requirements found."
    return False, output or f"`python -m pip check` exited with status {result.returncode}."


def _quickstart_check(source_root: Path, quickstart: Path) -> tuple[bool, str]:
    code = (
        "from sim.config import load_simulation_yaml, validate_scenario_plugins; "
        f"cfg=load_simulation_yaml({str(quickstart)!r}); "
        "errors=validate_scenario_plugins(cfg); "
        "assert not errors, '; '.join(errors[:3]); "
        "print('OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except Exception as exc:
        return False, f"isolated validation could not start: {exc}"
    lines = (result.stdout or result.stderr or "").strip().splitlines()
    detail = lines[-1] if lines else f"child process exited with status {result.returncode}"
    return result.returncode == 0, detail


def _write_check(output_root: Path) -> tuple[bool, str]:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".doctor_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True, str(output_root.resolve())
    except Exception as exc:
        return False, str(exc)


def remediation_commands(
    *,
    system: str | None = None,
    version_info: Sequence[int] | None = None,
) -> tuple[str, ...]:
    target = tuple(version_info or sys.version_info[:2])[:2]
    if not interpreter_is_supported(target):
        target = (3, 14)
    minor = f"{target[0]}.{target[1]}"
    constraint = f"constraints/py{target[0]}{target[1]}.txt"
    if str(system or platform.system()).lower() == "windows":
        win_constraint = constraint.replace("/", "\\")
        return (
            f"py -{minor} -m venv .venv",
            r".\.venv\Scripts\python.exe -m pip install --upgrade pip",
            rf'.\.venv\Scripts\python.exe -m pip install -c {win_constraint} ".[cross-platform]"',
            r".\.venv\Scripts\python.exe run_simulation.py --doctor",
        )
    return (
        f"python{minor} -m venv .venv",
        ".venv/bin/python -m pip install --upgrade pip",
        f'.venv/bin/python -m pip install -c {constraint} ".[cross-platform]"',
        ".venv/bin/python run_simulation.py --doctor",
    )


def _status_text(ok: bool, *, optional: bool = False, legacy: bool = False) -> str:
    if ok:
        return "OK"
    if legacy:
        return "LEGACY"
    return "UNAVAILABLE" if optional else "FAIL"


def _print_row(label: str, status: str, detail: str) -> None:
    print(f"{label:<22} : {status} - {detail}")


def _profile_detail(
    *,
    core_ok: bool,
    dev_ok: bool,
    game_ok: bool,
    accel_ok: bool,
    accel_qualified: bool,
    validation_ok: bool,
    ml_ok: bool,
) -> str:
    if not core_ok:
        return "incomplete core installation"
    capabilities = ["core"]
    if dev_ok:
        capabilities.append("dev")
    if game_ok:
        capabilities.append("game")
    if accel_ok:
        capabilities.append("accel")
    if validation_ok:
        capabilities.append("validation")
    if ml_ok:
        capabilities.append("ml")
    cross_platform_ok = dev_ok and game_ok and validation_ok and (accel_ok or not accel_qualified)
    if cross_platform_ok and ml_ok and accel_ok:
        profile = "full"
    elif cross_platform_ok:
        profile = "cross-platform"
    elif len(capabilities) == 1:
        profile = "core"
    else:
        profile = "custom"
    return f"{profile} ({', '.join(capabilities)})"


def print_doctor_report(*, source_root: str | Path | None = None) -> bool:
    root = Path(source_root).expanduser().resolve() if source_root is not None else Path(__file__).resolve().parents[1]
    try:
        quickstart = quickstart_config_path()
    except ResourceNotFoundError:
        quickstart = root / "configs" / "quickstart_5min.yaml"
    version = tuple(sys.version_info[:2])
    python_ok = interpreter_is_supported(version)
    security_ok, security_detail = security_support_detail(version)
    if version == (3, 10):
        security_ok = False
        security_detail += (
            " Pillow 12.3.0 is retained for functional compatibility only; known advisories block "
            "dependency security qualification, including core/media/full; no upstream 3.10 backport is available. "
            "Use CPython 3.11+ with Pillow 13."
        )
    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    platform_ok, platform_expected = _platform_is_supported(system, machine)

    print("")
    print("=" * 88)
    print("ORBITAL ENGINEERING LAB DOCTOR")
    print("=" * 88)
    print("Environment")
    _print_row(
        "Functional Python",
        _status_text(python_ok),
        f"{platform.python_implementation()} {platform.python_version()}; supported range is {SUPPORTED_PYTHON_RANGE}.",
    )
    _print_row(
        "Security baseline",
        _status_text(security_ok, legacy=python_ok and not security_ok),
        security_detail,
    )
    _print_row("Operating system", _status_text(platform_ok), f"{system}; expected {platform_expected}.")
    _print_row("OS version", "INFO", _os_version_detail(system))
    _print_row("Architecture", "INFO", machine)
    _print_row("Python executable", "INFO", str(Path(sys.executable).absolute()))
    version_status = inspect_project_version(source_root=root)
    _print_row(
        "OEL version",
        "OK" if version_status.ok else ("WARN" if not version_status.required else "FAIL"),
        version_status.detail,
    )
    disposition = os.environ.get("OEL_INSTALLATION_DISPOSITION", "developer").strip() or "developer"
    disposition_ok = disposition in {"official", "developer"}
    _print_row(
        "Installation state",
        _status_text(disposition_ok),
        f"{disposition}; use `oel update status --full` for content verification.",
    )
    workspace_root = os.environ.get("OEL_WORKSPACE_ROOT", "").strip()
    workspace_version = os.environ.get("OEL_ENGINE_VERSION", "").strip()
    _print_row(
        "Workspace pin",
        "INFO",
        f"{workspace_version or 'unmanaged'} at {workspace_root}" if workspace_root else "no managed workspace selected",
    )
    sys.stdout.flush()

    core_specs = _core_specs(version)
    dev_specs = _dev_specs(version)
    game_specs = _game_specs(version)
    all_specs = tuple(dict.fromkeys(core_specs + dev_specs + game_specs + ACCEL_SPECS + VALIDATION_SPECS + ML_SPECS))
    all_statuses = evaluate_dependencies(all_specs)
    status_by_name = {item.spec.distribution: item for item in all_statuses}

    def selected(specs: Iterable[DependencySpec]) -> tuple[DependencyStatus, ...]:
        return tuple(status_by_name[item.distribution] for item in specs)

    core_statuses = selected(core_specs)
    dev_statuses = selected(dev_specs)
    game_statuses = selected(game_specs)
    accel_statuses = selected(ACCEL_SPECS)
    validation_statuses = selected(VALIDATION_SPECS)
    ml_statuses = selected(ML_SPECS)
    core_ok = _all_compatible(core_statuses)
    dev_ok = _all_compatible(dev_statuses)
    game_ok = _all_compatible(game_statuses)
    accel_ok = _all_compatible(accel_statuses)
    validation_ok = _all_compatible(validation_statuses)
    ml_ok = _all_compatible(ml_statuses)
    accel_qualified = _acceleration_qualified(system, machine)

    print("-" * 88)
    print("Resolved packages")
    _print_row(
        "Core dependencies",
        _status_text(core_ok),
        "all required versions resolve" if core_ok else _failed_dependency_detail(core_statuses),
    )
    for item in all_statuses:
        required = item.spec in core_specs
        if required or item.version is not None:
            _print_row(
                f"Package {item.spec.distribution}",
                _status_text(item.compatible, optional=not required),
                item.detail,
            )
    pip_ok, pip_detail = _pip_check()
    _print_row("Dependency graph", _status_text(pip_ok), pip_detail)
    _print_row(
        "Install profile",
        "INFO",
        _profile_detail(
            core_ok=core_ok,
            dev_ok=dev_ok,
            game_ok=game_ok,
            accel_ok=accel_ok,
            accel_qualified=accel_qualified,
            validation_ok=validation_ok,
            ml_ok=ml_ok,
        ),
    )

    print("-" * 88)
    print("Capabilities")
    _print_row(
        "Core runtime",
        _status_text(core_ok),
        "CLI, YAML/API, plotting, and review dependencies are compatible."
        if core_ok
        else _failed_dependency_detail(core_statuses),
    )
    dev_available, dev_detail = _capability_detail(
        dev_statuses,
        install_extra="dev",
        available_detail="pytest and Ruff development tooling are available.",
    )
    _print_row("Development/test", _status_text(dev_available, optional=True), dev_detail)
    game_available, game_detail = _capability_detail(
        game_statuses,
        install_extra="game",
        available_detail="Pygame trainer and media dependencies are available.",
    )
    _print_row("Trainer", _status_text(game_available, optional=True), game_detail)
    if not accel_qualified and not accel_ok:
        accel_detail = (
            "not universally qualified on macOS x86_64 because approved Numba wheels are unavailable; "
            "deterministic serial execution remains available."
        )
    else:
        _accel_available, accel_detail = _capability_detail(
            accel_statuses,
            install_extra="accel",
            available_detail="Numba acceleration dependency is available.",
        )
    _print_row("Acceleration", _status_text(accel_ok, optional=True), accel_detail)
    validation_available, validation_detail = _capability_detail(
        validation_statuses,
        install_extra="validation",
        available_detail="OGP/SGP4 validation dependency is available.",
    )
    _print_row("OGP validation", _status_text(validation_available, optional=True), validation_detail)
    ml_available, ml_detail = _capability_detail(
        ml_statuses,
        install_extra="ml",
        available_detail="Separately qualified machine-learning dependencies are available.",
    )
    _print_row("Machine learning", _status_text(ml_available, optional=True), ml_detail)

    print("-" * 88)
    print("Runtime readiness")
    quickstart_exists = quickstart.is_file()
    _print_row(
        "Quickstart config",
        _status_text(quickstart_exists),
        str(quickstart),
    )
    output_root = Path(os.environ.get("OEL_OUTPUT_ROOT", str(root / "outputs"))).expanduser().resolve()
    output_ok, output_detail = _write_check(output_root)
    _print_row("Output directory", _status_text(output_ok), output_detail)
    prerequisite_ok = python_ok and platform_ok and core_ok and pip_ok and quickstart_exists
    if prerequisite_ok:
        quickstart_ok, quickstart_detail = _quickstart_check(root, quickstart)
    else:
        quickstart_ok = False
        quickstart_detail = "not attempted until the failed interpreter, platform, or core dependency checks are repaired."
    _print_row("Quickstart validation", _status_text(quickstart_ok), quickstart_detail)

    overall_ok = (
        python_ok
        and platform_ok
        and core_ok
        and pip_ok
        and (version_status.ok or not version_status.required)
        and output_ok
        and quickstart_ok
    )
    print("-" * 88)
    print("Recovery commands")
    for command in remediation_commands(system=system, version_info=version):
        print(f"  {command}")
    print("-" * 88)
    if overall_ok:
        print("READY: the functional core is ready. Optional capability status is reported above.")
    else:
        print(
            "NOT READY: repair each FAIL item with the commands above, then rerun "
            "`run_simulation.py --doctor`."
        )
    print("=" * 88)
    return overall_ok


def require_supported_interpreter() -> None:
    if interpreter_is_supported():
        return
    version = platform.python_version()
    print(
        f"OEL cannot start with CPython {version}. The declared functional range is "
        f"{SUPPORTED_PYTHON_RANGE}.",
        file=sys.stderr,
    )
    print(
        "Run `python run_simulation.py --doctor` for OS-appropriate recovery commands.",
        file=sys.stderr,
    )
    raise SystemExit(2)


__all__ = [
    "DOCTOR_CONTRACT_LABELS",
    "DependencySpec",
    "DependencyStatus",
    "SUPPORTED_PYTHON_MAX_EXCLUSIVE",
    "SUPPORTED_PYTHON_MIN",
    "SUPPORTED_PYTHON_RANGE",
    "dependency_is_compatible",
    "doctor_requested",
    "evaluate_dependencies",
    "interpreter_is_supported",
    "print_doctor_report",
    "remediation_commands",
    "require_supported_interpreter",
    "security_support_detail",
]
