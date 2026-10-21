from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from sim.doctor import (
    CORE_SPECS,
    DependencySpec,
    dependency_is_compatible,
    evaluate_dependencies,
    interpreter_is_supported,
    remediation_commands,
    security_support_detail,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("version", [(3, 10), (3, 11), (3, 12), (3, 13), (3, 14)])
def test_doctor_accepts_exact_declared_python_minors(version: tuple[int, int]) -> None:
    assert interpreter_is_supported(version)


@pytest.mark.parametrize("version", [(3, 9), (3, 15), (4, 0)])
def test_doctor_rejects_python_outside_bounded_range(version: tuple[int, int]) -> None:
    assert not interpreter_is_supported(version)


def test_security_baseline_is_distinct_from_functional_compatibility() -> None:
    assert interpreter_is_supported((3, 10))
    supported, current_detail = security_support_detail((3, 10), today=date(2026, 7, 26))
    legacy, legacy_detail = security_support_detail((3, 10), today=date(2026, 11, 1))

    assert supported is True
    assert "through 2026-10" in current_detail
    assert legacy is False
    assert "functional legacy tier" in legacy_detail


def test_dependency_compatibility_reports_missing_and_out_of_range_versions() -> None:
    specs = (
        DependencySpec("numpy", "2.1", "2.5"),
        DependencySpec("ruff", exact="0.15.14"),
        DependencySpec("missing", "1", "2"),
    )
    statuses = evaluate_dependencies(
        specs,
        installed_versions={
            "numpy": "1.26.4",
            "ruff": "0.15.14",
            "missing": None,
        },
    )

    assert statuses[0].compatible is False
    assert "requires >=2.1,<2.5" in statuses[0].detail
    assert statuses[1].compatible is True
    assert statuses[2].version is None
    assert "missing" in statuses[2].detail
    numpy_spec = next(spec for spec in CORE_SPECS if spec.distribution == "numpy")
    assert dependency_is_compatible("2.4.6", numpy_spec)
    assert not dependency_is_compatible("2.5.0", numpy_spec)


def test_doctor_prints_windows_recovery_commands() -> None:
    commands = remediation_commands(system="Windows", version_info=(3, 13))

    assert commands == (
        "py -3.13 -m venv .venv",
        r".\.venv\Scripts\python.exe -m pip install --upgrade pip",
        r'.\.venv\Scripts\python.exe -m pip install -c constraints\py313.txt ".[cross-platform]"',
        r".\.venv\Scripts\python.exe run_simulation.py --doctor",
    )


def test_doctor_uses_latest_supported_minor_to_recover_unsupported_python() -> None:
    commands = remediation_commands(system="Linux", version_info=(3, 15))

    assert commands[0] == "python3.14 -m venv .venv"
    assert "constraints/py314.txt" in commands[2]
    assert all("3.10+" not in command for command in commands)


def test_doctor_runs_without_site_packages_and_reports_recovery() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "run_simulation.py", "--doctor"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ORBITAL ENGINEERING LAB DOCTOR" in result.stdout
    assert "Core dependencies" in result.stdout
    assert "missing distributions/wheels" in result.stdout
    assert "Quickstart validation  : FAIL - not attempted" in result.stdout
    assert "Recovery commands" in result.stdout
    assert "Traceback" not in result.stderr


def test_doctor_bootstrap_precedes_heavy_runtime_imports() -> None:
    source = (ROOT / "run_simulation.py").read_text(encoding="utf-8")

    doctor_guard = source.index("if doctor_requested(sys.argv[1:])")
    first_heavy_import = min(
        source.index(import_line)
        for import_line in (
            "from sim.acceleration.settings import",
            "from sim.config import",
        )
        if import_line in source
    )
    assert doctor_guard < first_heavy_import


def test_doctor_warns_legacy_pillow_security_without_removing_python310_function(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    import sim.doctor as doctor

    assert interpreter_is_supported((3, 10))
    monkeypatch.setattr(doctor.sys, "version_info", (3, 10))
    monkeypatch.setattr(doctor, "_installed_version", lambda _distribution: None)
    monkeypatch.setattr(doctor, "_pip_check", lambda: (True, "fixture dependency graph"))
    monkeypatch.setattr(doctor, "_write_check", lambda _root: (True, "fixture output directory"))
    doctor.print_doctor_report(source_root=tmp_path)
    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if "Security baseline" in line)
    assert "LEGACY" in row
    assert "known advisories block dependency security qualification, including core/media/full" in row
    assert "CPython 3.11+ with Pillow 13" in row
    assert "Functional Python" in output
