from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_authoritative_installation_guide_covers_supported_platforms_and_minors() -> None:
    guide = _read("docs/installation.md")

    windows_commands = [
        "py --list",
        "py -0p",
        "py -3.14 -m venv .venv",
        r".\.venv\Scripts\python.exe -m pip install --upgrade pip",
        r'.\.venv\Scripts\python.exe -m pip install -c constraints\py314.txt --find-links dist/native ".[dev]"',
        r".\.venv\Scripts\python.exe run_simulation.py --doctor",
        r".\.venv\Scripts\python.exe run_simulation.py --quickstart",
    ]
    posix_commands = [
        "python3.14 -m venv .venv",
        ".venv/bin/python -m pip install --upgrade pip",
        '.venv/bin/python -m pip install -c constraints/py314.txt --find-links dist/native ".[dev]"',
        ".venv/bin/python run_simulation.py --doctor",
        ".venv/bin/python run_simulation.py --quickstart",
    ]

    for command in windows_commands + posix_commands:
        assert command in guide
    for minor in range(10, 15):
        assert f"3.{minor}" in guide
        assert f"constraints/py3{minor}.txt" in guide

    assert "## Windows PowerShell" in guide
    assert "## macOS Or Linux (POSIX Shell)" in guide
    assert "## Classroom Or Restricted Environment Check" in guide
    assert "## Troubleshooting A Failed Installation" in guide
    assert "Activation is optional" in guide


def test_offline_installation_guide_matches_v1_single_local_edition() -> None:
    guide = _read("docs/offline-installation.md")

    assert "The v1 local edition does not require a product license or activation." in guide
    assert "For Pro bundles" not in guide
    assert "offline license" not in guide
    assert "does not replace or initialize the external trust anchor" in guide


def test_entry_onboarding_docs_keep_explicit_windows_and_posix_paths() -> None:
    onboarding_paths = ["README.md", "docs/quickstart.md"]
    if (ROOT / "docs" / "public-readme.md").is_file():
        onboarding_paths.append("docs/public-readme.md")
    for relative_path in onboarding_paths:
        text = _read(relative_path)
        assert "docs/installation.md" in text or "(installation.md)" in text
        assert "```powershell" in text
        assert "py --list" in text
        assert r".\.venv\Scripts\python.exe" in text
        assert "python3.14 -m venv .venv" in text
        assert ".venv/bin/python" in text

    contributor_paths = ["CONTRIBUTING.md"]
    if (ROOT / "docs" / "pro-user-guide.md").is_file():
        contributor_paths.append("docs/pro-user-guide.md")
    for relative_path in contributor_paths:
        text = _read(relative_path)
        assert "installation.md" in text
        assert "```powershell" in text
        assert r".\.venv\Scripts\python.exe" in text
        assert "python3.14 -m venv .venv" in text
        assert "source .venv/bin/activate" in text


def test_public_readme_leads_with_the_managed_release_installation() -> None:
    # The export installs this maintained source as its root README.
    readme = _read("docs/public-readme.md" if (ROOT / "docs/public-readme.md").is_file() else "README.md")

    managed_heading = "## Install And Run"
    contributor_heading = "## Source Installation For Contributors"
    posix_installer = (
        "https://github.com/adamcohen8/orbital-engineering-lab/"
        "releases/latest/download/install.sh"
    )
    powershell_installer = (
        "https://github.com/adamcohen8/orbital-engineering-lab/"
        "releases/latest/download/install.ps1"
    )
    managed_commands = [
        "curl --proto '=https' --tlsv1.2 -fsSLo /tmp/oel-install.sh",
        "less /tmp/oel-install.sh",
        "sh /tmp/oel-install.sh",
        "Invoke-WebRequest",
        "Get-Content $env:TEMP\\oel-install.ps1",
        "oel update status --full",
        "oel doctor",
        "oel workspace init my-oel-workspace",
        "oel --workspace my-oel-workspace sim --quickstart --validate-only",
        "oel --workspace my-oel-workspace sim --quickstart",
    ]

    for expected in (posix_installer, powershell_installer, *managed_commands):
        assert expected in readme
    assert "does not edit or overwrite a workspace" in readme
    assert readme.index(managed_heading) < readme.index(contributor_heading)
    assert readme.index(posix_installer) < readme.index("git clone https://github.com/")


def test_general_and_agent_docs_use_the_portable_activated_command_convention() -> None:
    general_docs = [
        "SECURITY.md",
        "docs/security/supply-chain.md",
        "docs/known-limitations.md",
        "docs/compatibility.md",
        "AGENTS.md",
        "agents/public/AGENTS.md",
        "docs/oel-agents.md",
        "docs/agent-capability-routing.md",
        "docs/agent-task-runner.md",
        "docs/agent-task-cards.md",
    ]
    for relative_path in general_docs:
        text = _read(relative_path)
        assert ".venv/bin/python" not in text, relative_path
        assert "installation.md" in text, relative_path

    for card_path in sorted((ROOT / "agents" / "tasks").glob("*.md")):
        text = card_path.read_text(encoding="utf-8")
        assert "../../docs/installation.md" in text, card_path
        assert ".venv/bin/python" not in text, card_path
        assert "python " in text, card_path


def test_bug_report_and_generated_output_commands_are_cross_platform() -> None:
    bug_report = _read(".github/ISSUE_TEMPLATE/bug_report.yml")
    output_index = _read("sim/reporting/output_index.py")

    assert r".\.venv\Scripts\python.exe --version" in bug_report
    assert ".venv/bin/python --version" in bug_report
    assert "Install profile and constraints:" in bug_report
    assert ".venv/bin/python" not in output_index
    assert "python -m sim.review" in output_index
    assert "docs/installation.md" in output_index


def test_installation_guide_and_python_api_builder_are_public_owned() -> None:
    task_card = _read("agents/tasks/python_api_minimal_propagation.md")
    manifest_path = ROOT / "docs" / "operations" / "public_surface_manifest.yaml"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public_onboarding = manifest["public_surfaces"]["public onboarding docs"]["required"]
        assert "docs/installation.md" in public_onboarding
    else:
        assert (ROOT / "docs" / "installation.md").is_file()
    assert "python agents/examples/build_public_agent_python_api_minimal_propagation.py" in task_card
    assert (ROOT / "agents" / "examples" / "build_public_agent_python_api_minimal_propagation.py").is_file()
