from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from sim.doctor import CORE_SPECS, SUPPORTED_PYTHON_RANGE, _platform_is_supported
from tools.compatibility_matrix import build_matrix
from tools.run_compatibility_smoke import (
    DESKTOP_ACCEPTANCE,
    HOSTED_ACCEPTANCE,
    LOCAL_ACCEPTANCE,
    _github_hosted_provenance,
    _load_desktop_attestation,
    _parser,
    _validate_audit_result,
)

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_MINORS = ("3.10", "3.11", "3.12", "3.13", "3.14")
CROSS_PLATFORM_EXTRAS = ("dev", "game", "accel", "validation", "cross-platform")


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _constraint_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"{path.name}: expected an approved exact version: {line}"
        names.add(line.split("==", 1)[0].strip().lower())
    return names


def test_python_range_classifiers_and_os_classifiers_match_phase2_contract() -> None:
    project = _pyproject()["project"]

    assert project["requires-python"] == ">=3.10,<3.15"
    assert SUPPORTED_PYTHON_RANGE == project["requires-python"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE.txt"]
    classifiers = set(project["classifiers"])
    assert not any(item.startswith("License ::") for item in classifiers)
    for minor in SUPPORTED_MINORS:
        assert f"Programming Language :: Python :: {minor}" in classifiers
    assert {
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    }.issubset(classifiers)


def test_doctor_core_dependency_ranges_match_package_metadata() -> None:
    from packaging.requirements import Requirement

    project_requirements = {
        str(item).split(";", 1)[0].replace(" ", "").lower()
        for item in _pyproject()["project"]["dependencies"]
        for requirement in [Requirement(item)]
        if requirement.marker is None or requirement.marker.evaluate()
    }
    doctor_requirements = {f"{spec.distribution}{spec.requirement}".lower() for spec in CORE_SPECS}

    assert doctor_requirements == project_requirements


def test_numpy_2_migration_lint_gate_is_enabled() -> None:
    ruff_select = set(_pyproject()["tool"]["ruff"]["lint"]["select"])
    assert "NPY201" in ruff_select


def test_direct_dependencies_and_supported_extras_are_bounded() -> None:
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]

    for requirement in project["dependencies"]:
        assert "<" in requirement, requirement
    for extra in CROSS_PLATFORM_EXTRAS + ("ml", "full"):
        assert extra in extras
        for requirement in extras[extra]:
            assert "<" in requirement or "==" in requirement, f"{extra}: {requirement}"

    assert any(item.startswith("pygame") and "python_version < '3.14'" in item for item in extras["game"])
    assert any(item.startswith("pygame-ce") and "python_version >= '3.14'" in item for item in extras["game"])
    for profile in ("game", "cross-platform", "full"):
        assert any(item.startswith("oel-rust-game>=") for item in extras[profile])
        assert any(item.startswith("oel-rust-orbit>=") for item in extras[profile])
    assert any(
        item.startswith("numba") and "platform_system != 'Darwin'" in item and "platform_machine != 'x86_64'" in item
        for item in extras["cross-platform"]
    )
    assert any(item.startswith("tomli") and "python_version < '3.11'" in item for item in extras["dev"])
    assert any(item.startswith("setuptools") for item in extras["dev"])
    assert not any("torch" in item.lower() for item in extras["cross-platform"])
    assert "torch>=2.13,<2.14" in extras["ml"]
    assert "torch>=2.13,<2.14" in extras["full"]


def test_every_supported_python_minor_has_an_approved_constraint_set() -> None:
    expected_packages = {
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pytest",
        "ruff",
        "setuptools",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "numba",
        "llvmlite",
        "sgp4",
    }
    for minor in SUPPORTED_MINORS:
        path = ROOT / "constraints" / f"py{minor.replace('.', '')}.txt"
        assert path.is_file()
        assert expected_packages.issubset(_constraint_names(path))

    assert "pygame" in _constraint_names(ROOT / "constraints" / "py313.txt")
    assert "pygame-ce" in _constraint_names(ROOT / "constraints" / "py314.txt")
    assert "tomli" in _constraint_names(ROOT / "constraints" / "py310.txt")


def test_profile_definitions_and_reference_ci_use_constraints_and_dependency_evidence() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "cross-platform" in readme
    assert "docs/installation.md" in readme

    for rel_path in ("docs/compatibility.md", "docs/security/supply-chain.md"):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "cross-platform" in text, rel_path
        assert "constraints/py311.txt" in text, rel_path

    for rel_path in ("docs/public-readme.md", "docs/pro-user-guide.md"):
        path = ROOT / rel_path
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "cross-platform" in text, rel_path
            assert "constraints/py314.txt" in text, rel_path
            assert "installation.md" in text, rel_path

    workflow = (ROOT / "tools" / "run_supply_chain_gate.py").read_text(encoding="utf-8")
    for artifact in (
        "pip-install-report.json",
        "pip-check.txt",
        "python-freeze.txt",
        "wheel-inventory.json",
        "sbom.cdx.json",
        "pip-audit.json",
    ):
        assert artifact in workflow


def test_public_trainer_guides_include_explicit_windows_and_posix_commands() -> None:
    public_readme = "docs/public-readme.md" if (ROOT / "docs/public-readme.md").is_file() else "README.md"
    for rel_path in (public_readme, "docs/rpo-trainer-instructor-one-pager.md"):
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "Windows PowerShell" in text, rel_path
        assert r".\.venv\Scripts\python.exe -m pip install" in text, rel_path
        assert r".\.venv\Scripts\python.exe run_game.py" in text, rel_path
        assert ".venv/bin/python -m pip install" in text, rel_path
        assert ".venv/bin/python run_game.py" in text, rel_path
        assert 'constraints/py314.txt ".[game]"' in text, rel_path
        assert "run_simulation.py --doctor" in text, rel_path


def test_compatibility_workflow_runs_one_unavailable_host_row() -> None:
    path = ROOT / ".github" / "workflows" / "compatibility.yml"
    source = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    jobs = workflow["jobs"]
    smoke = jobs["platform-diagnostic"]
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert set(jobs) == {"platform-diagnostic"}
    assert "strategy" not in smoke
    assert smoke["timeout-minutes"] == 20
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["target"]["options"] == ["windows-x64", "macos-intel"]
    assert inputs["python-version"]["options"] == list(SUPPORTED_MINORS)

    for required_text in (
        "--only-binary=:all:",
        "pip-install-report.json",
        "run_compatibility_smoke.py",
        "--acceptance-class github-hosted-automation",
        "--skip-dependency-audit",
        "windows-2022",
        "macos-15-intel",
        "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f # v6",
    ):
        assert required_text in source
    for prohibited_text in (
        "schedule:",
        "matrix:",
        "pip_audit",
        "pytest",
        "validation/compatibility_acceptance.py",
    ):
        assert prohibited_text not in source


def test_local_matrix_definition_still_covers_declared_compatibility_program() -> None:
    full_rows = build_matrix("full")["include"]
    canary_rows = build_matrix("canary")["include"]
    assert len(full_rows) == 20
    assert len(canary_rows) == 14
    assert {row["runner"] for row in full_rows} == {
        "ubuntu-22.04",
        "windows-2022",
        "macos-15",
        "macos-15-intel",
    }
    assert {row["python_version"] for row in full_rows} == set(SUPPORTED_MINORS)
    boundary_rows = [row for row in full_rows if row["boundary"]]
    assert len(boundary_rows) == 8
    assert {row["python_version"] for row in boundary_rows} == {"3.10", "3.14"}
    assert len({row["runner"] for row in boundary_rows}) == 4


def test_public_export_does_not_generate_hosted_test_workflows() -> None:
    generator = (ROOT / "tools" / "public_export" / "workflows.py").read_text(encoding="utf-8")
    assert "PUBLIC_PAGES_WORKFLOW" in generator
    assert "PUBLIC_COMPATIBILITY_WORKFLOW" not in generator
    assert "PUBLIC_CI_WORKFLOW" not in generator
    assert "PUBLIC_DEPENDENCY_AUDIT_WORKFLOW" not in generator


def test_desktop_evidence_requires_explicit_manual_attestations(tmp_path: Path) -> None:
    attestation = tmp_path / "desktop-attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "native_folder_open_verified": True,
                "trainer_window_verified": True,
                "keyboard_input_verified": False,
                "display_rendering_verified": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keyboard_input_verified"):
        _load_desktop_attestation(attestation)


def test_desktop_and_hosted_evidence_classes_cannot_be_confused(tmp_path: Path) -> None:
    attestation = tmp_path / "desktop-attestation.json"
    payload = {
        "native_folder_open_verified": True,
        "trainer_window_verified": True,
        "keyboard_input_verified": True,
        "display_rendering_verified": True,
    }
    attestation.write_text(json.dumps(payload), encoding="utf-8")

    with (
        patch("tools.run_compatibility_smoke.platform.system", return_value="Windows"),
        patch("tools.run_compatibility_smoke.platform.release", return_value="11"),
        patch("tools.run_compatibility_smoke.platform.machine", return_value="AMD64"),
    ):
        assert _load_desktop_attestation(attestation) == payload

    assert DESKTOP_ACCEPTANCE == "controlled-windows-11-desktop"
    assert HOSTED_ACCEPTANCE == "github-hosted-automation"
    assert LOCAL_ACCEPTANCE == "local-diagnostic"
    assert _parser().parse_args(
        [
            "--constraints",
            "constraints/py311.txt",
            "--install-report",
            "install.json",
            "--audit-result",
            "audit.json",
            "--output-dir",
            "evidence",
        ]
    ).acceptance_class == LOCAL_ACCEPTANCE

    hosted = _parser().parse_args(
        [
            "--constraints",
            "constraints/py311.txt",
            "--install-report",
            "install.json",
            "--skip-dependency-audit",
            "--output-dir",
            "evidence",
            "--acceptance-class",
            HOSTED_ACCEPTANCE,
        ]
    )
    assert hosted.skip_dependency_audit is True


def test_hosted_evidence_requires_github_actions_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "GITHUB_ACTIONS",
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_SHA",
        "RUNNER_OS",
        "RUNNER_ARCH",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="GitHub-hosted evidence requires"):
        _github_hosted_provenance()

    values = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_WORKFLOW": "Cross-platform compatibility",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "a" * 40,
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    assert _github_hosted_provenance()["run_id"] == "123"


def test_compatibility_packet_rejects_audit_from_a_different_environment(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "pip-audit.json"
    audit.write_text(
        json.dumps(
            {
                "dependencies": [
                    {"name": "NumPy", "version": "2.4.6", "vulns": []},
                    {"name": "setuptools", "version": "80.10.2", "vulns": []},
                ],
                "fixes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="setuptools"):
        _validate_audit_result(
            audit,
            installed_versions={"numpy": "2.4.6", "setuptools": "83.0.0"},
        )


def test_compatibility_packet_rejects_incomplete_or_vulnerable_audit(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps(
            {
                "dependencies": [{"name": "numpy", "version": "2.4.6", "vulns": []}],
                "fixes": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="complete installed environment"):
        _validate_audit_result(
            incomplete,
            installed_versions={"numpy": "2.4.6", "scipy": "1.17.1"},
        )

    vulnerable = tmp_path / "vulnerable.json"
    vulnerable.write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "numpy",
                        "version": "2.4.6",
                        "vulns": [{"id": "DEMO-CVE", "fix_versions": ["2.4.7"]}],
                    }
                ],
                "fixes": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEMO-CVE"):
        _validate_audit_result(
            vulnerable,
            installed_versions={"numpy": "2.4.6"},
        )


@pytest.mark.parametrize("project_name", ("orbital-engineering-lab", "orbital-engagement-lab"))
def test_compatibility_packet_allows_only_the_local_first_party_project_to_be_unaudited(
    tmp_path: Path,
    project_name: str,
) -> None:
    audit = tmp_path / "pip-audit.json"
    audit.write_text(
        json.dumps(
            {
                "dependencies": [{"name": "numpy", "version": "2.4.6", "vulns": []}],
                "fixes": [],
            }
        ),
        encoding="utf-8",
    )

    payload = _validate_audit_result(
        audit,
        installed_versions={
            "numpy": "2.4.6",
            project_name: "0.23.1",
        },
    )

    assert payload["dependencies"][0]["name"] == "numpy"


def test_doctor_accepts_windows_server_2022_only_as_the_automation_baseline() -> None:
    with patch("sim.doctor.platform.version", return_value="10.0.20348"):
        supported, expected = _platform_is_supported("Windows", "AMD64")
    assert supported
    assert "Windows Server 2022 x64 automation" in expected

    with patch("sim.doctor.platform.version", return_value="10.0.17763"):
        supported, _ = _platform_is_supported("Windows", "AMD64")
    assert not supported


@pytest.mark.parametrize("minor", SUPPORTED_MINORS)
def test_pillow_requirement_preserves_legacy_function_and_patched_media_baseline(minor: str) -> None:
    from packaging.requirements import Requirement

    from sim.doctor import _core_specs, _game_specs, dependency_is_compatible

    version = tuple(int(part) for part in minor.split("."))
    for profile in ("core", "game", "cross-platform", "full"):
        project = _pyproject()["project"]
        requirements = project["dependencies"] if profile == "core" else project["optional-dependencies"][profile]
        active = [
            requirement for item in requirements
            for requirement in [Requirement(item)]
            if requirement.name.lower() == "pillow"
            and (requirement.marker is None or requirement.marker.evaluate({"python_version": minor}))
        ]
        assert len(active) == 1
        expected = "12.3.0" if version == (3, 10) else "13.0.0"
        other = "13.0.0" if version == (3, 10) else "12.3.0"
        assert expected in active[0].specifier
        assert other not in active[0].specifier
        specs = _core_specs(version) if profile == "core" else _game_specs(version)
        spec = next(row for row in specs if row.distribution == "Pillow")
        assert dependency_is_compatible(expected, spec)
        assert not dependency_is_compatible(other, spec)
        constraints = (ROOT / "constraints" / f"py{minor.replace('.', '')}.txt").read_text()
        assert f"pillow=={expected}" in constraints
    if version == (3, 10):
        assert "security qualification (including core/media/full) is blocked" in " ".join(
            (ROOT / "docs" / "compatibility.md").read_text().split()
        )
