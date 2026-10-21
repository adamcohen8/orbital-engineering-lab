from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_installable_release import _source_files, release_source_identity  # noqa: E402
from tools.generate_python_sbom import write_sbom  # noqa: E402
from tools.native_source_identity import native_source_identity  # noqa: E402

PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX_URL = "https://pypi.org/simple"
PIP_VERSION = "26.2.1"
PIP_AUDIT_VERSION = "2.10.1"
_NATIVE_WHEEL_PACKAGES = (
    ("oel-rust-orbit", "oel_rust_orbit", "oel-orbit"),
    ("oel-rust-game", "oel_rust_game", "oel-game"),
)


def _package_version(source_root: Path | None = None) -> str:
    text = ((source_root or ROOT) / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("pyproject.toml does not declare a package version")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, Any]:
    def capture(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _package_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("PIP_")}
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_INDEX_URL": PYPI_INDEX_URL,
        }
    )
    return environment


def _run(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, check=False)



def _normalized_distribution_name(value: object) -> str:
    return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())


def _native_qualification_manifest(source_root: Path | None = None) -> Path:
    version = _package_version() if source_root is None else _package_version(source_root)
    return (source_root or ROOT) / "rust" / f"qualification-v{version}.json"


def _qualified_native_wheel_receipts(
    *, python_minor: tuple[int, int] | None = None,
    source_root: Path | None = None, qualification_manifest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the current-source native wheel receipts for this host/interpreter."""
    manifest = qualification_manifest or _native_qualification_manifest(source_root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    version = _package_version() if source_root is None else _package_version(source_root)
    if payload.get("release_version") != version:
        raise ValueError("Native qualification release version does not match")
    major, minor = python_minor or (sys.version_info.major, sys.version_info.minor)
    prefix = f"{major}.{minor}."
    reports = [
        row for row in payload.get("reports", [])
        if isinstance(row, dict)
        and row.get("status") == "passed"
        and row.get("system") == platform.system()
        and row.get("machine") == platform.machine()
        and str(row.get("python", "")).startswith(prefix)
    ]
    selected: dict[str, dict[str, Any]] = {}
    for distribution, module_name, crate_name in _NATIVE_WHEEL_PACKAGES:
        matches = [row for row in reports if str(row.get("wheel", "")).startswith(module_name + "-")]
        if len(matches) != 1:
            raise ValueError(f"Native wheel needs one passing host/interpreter qualification: {module_name}")
        row = matches[0]
        crate = (source_root or ROOT) / "rust" / crate_name
        source_identity = native_source_identity(crate)
        if row.get("source_identity") != source_identity:
            raise ValueError(f"Native wheel qualification differs from candidate source: {module_name}")
        name = str(row.get("wheel", ""))
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("Native qualification contains an invalid wheel name")
        digest = str(row.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Native qualification has an invalid wheel SHA-256: {name}")
        selected[distribution] = {
            "wheel": name,
            "sha256": digest,
            "source_identity_sha256": source_identity["sha256"],
        }
    return selected


def _qualified_native_wheels(
    wheelhouse: Path,
    *,
    qualified_receipts: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, ...]:
    """Accept only bytes qualified for this release, host and Python minor."""
    house = wheelhouse.expanduser().resolve()
    receipts = qualified_receipts if qualified_receipts is not None else _qualified_native_wheel_receipts()
    selected = []
    for distribution, _module_name, _crate_name in _NATIVE_WHEEL_PACKAGES:
        receipt = receipts.get(distribution)
        if receipt is None:
            raise ValueError(f"Native wheel has no current qualification receipt: {distribution}")
        name = str(receipt["wheel"])
        wheel = house / name
        if wheel.is_symlink() or not wheel.is_file() or _sha256(wheel) != receipt.get("sha256"):
            raise ValueError(f"Native wheel differs from qualified bytes: {name}")
        selected.append(wheel)
    return tuple(selected)


def _full_install_command(
    *,
    python_executable: str,
    constraints: Path,
    install_report_path: Path,
    torch_cpu_index: bool,
    native_wheels: tuple[Path, ...] = (),
    source_root: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--force-reinstall",
        "--index-url",
        PYPI_INDEX_URL,
        "-c",
        str(constraints),
    ]
    if torch_cpu_index:
        command.extend(["--extra-index-url", PYTORCH_CPU_INDEX_URL])
    project = ".[full]" if source_root is None else str(source_root) + "[full]"
    command.extend([*(str(path) for path in native_wheels), project, "--report", str(install_report_path)])
    return command


def _build_requirements(source_root: Path | None = None) -> tuple[str, ...]:
    payload = tomllib.loads(((source_root or ROOT) / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = tuple(str(item).strip() for item in payload.get("build-system", {}).get("requires", []))
    if not requirements or any(not item for item in requirements):
        raise RuntimeError("pyproject.toml must declare non-empty build-system requirements")
    return requirements


def _build_dependency_install_command(
    *,
    python_executable: str,
    constraints: Path,
    install_report_path: Path,
    source_root: Path | None = None,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--force-reinstall",
        "--index-url",
        PYPI_INDEX_URL,
        "-c",
        str(constraints),
        *_build_requirements(source_root),
        "--report",
        str(install_report_path),
    ]


def _venv_python(environment_root: Path) -> Path:
    if sys.platform == "win32":
        return environment_root / "Scripts" / "python.exe"
    return environment_root / "bin" / "python"


def _recorded_run(
    command_results: list[dict[str, Any]],
    cmd: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    proc = _run(cmd, env=env)
    command_results.append({"command": cmd, "return_code": int(proc.returncode)})
    return proc


def _validate_dependency_sources(
    path: Path,
    *,
    torch_cpu_index: bool,
    native_wheels: tuple[Path, ...] = (),
    qualified_native_wheels: dict[str, dict[str, Any]] | None = None,
    source_root: Path | None = None,
) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_hosts = {"files.pythonhosted.org", "pypi.org"}
    if torch_cpu_index:
        allowed_hosts.add("download.pytorch.org")
    native_hashes = {wheel.name: _sha256(wheel) for wheel in native_wheels}
    seen_native_packages: set[str] = set()
    seen_local_native_artifacts: set[str] = set()
    native_package_counts: dict[str, int] = {}
    unexpected: list[str] = []
    for row in list(dict(payload).get("packages", []) or []):
        if not isinstance(row, dict):
            continue
        distribution = _normalized_distribution_name(row.get("name"))
        is_native = distribution in {item[0] for item in _NATIVE_WHEEL_PACKAGES}
        if is_native and qualified_native_wheels is not None:
            native_package_counts[distribution] = native_package_counts.get(distribution, 0) + 1
            if native_package_counts[distribution] > 1:
                unexpected.append(f"duplicate-native:{distribution}")
        source_url = str(row.get("source_url", "") or "")
        if source_url == "<local-source>":
            if row.get("name") == "orbital-engineering-lab" and row.get("version") == _package_version(source_root):
                continue
            artifact = str(row.get("artifact", ""))
            if is_native and qualified_native_wheels is not None:
                receipt = qualified_native_wheels.get(distribution)
                matches_receipt = (
                    receipt is not None
                    and row.get("artifact_type") == "wheel"
                    and artifact == receipt.get("wheel")
                    and row.get("sha256") == receipt.get("sha256")
                )
                matches_input = not native_hashes or (
                    artifact in native_hashes and row.get("sha256") == native_hashes[artifact]
                )
                if matches_receipt and matches_input:
                    seen_native_packages.add(distribution)
                    if native_hashes:
                        seen_local_native_artifacts.add(artifact)
                    continue
                unexpected.append(f"unqualified-native:{distribution}")
                continue
            if artifact in native_hashes and row.get("sha256") == native_hashes[artifact]:
                seen_local_native_artifacts.add(artifact)
                seen_native_packages.add(distribution)
                continue
            unexpected.append(f"unqualified-local:{row.get('name', '')}")
            continue
        if urlparse(source_url).hostname not in allowed_hosts:
            unexpected.append(source_url or "<missing-source-url>")
            continue
        if is_native and qualified_native_wheels is not None:
            receipt = qualified_native_wheels.get(distribution)
            artifact = str(row.get("artifact", ""))
            matches_receipt = (
                receipt is not None
                and row.get("artifact_type") == "wheel"
                and artifact == receipt.get("wheel")
                and row.get("sha256") == receipt.get("sha256")
            )
            if matches_receipt:
                seen_native_packages.add(distribution)
            else:
                unexpected.append(f"unqualified-native:{distribution}")
    if native_hashes and set(native_hashes) != seen_local_native_artifacts:
        unexpected.append("missing-qualified-native-wheel")
    if qualified_native_wheels is not None:
        expected_native_packages = set(qualified_native_wheels)
        if seen_native_packages != expected_native_packages:
            missing = sorted(expected_native_packages - seen_native_packages)
            unexpected.extend(f"missing-qualified-native:{name}" for name in missing)
    if unexpected:
        return "Dependency evidence contains unapproved or unqualified sources: " + ", ".join(sorted(set(unexpected)))
    return None


def _run_supply_chain_gate_in_environment(
    output: Path,
    *,
    install_full: bool,
    bootstrap_python: str,
    constraints: Path,
    torch_cpu_index: bool,
    isolated_environment_root: Path | None,
    native_wheels: tuple[Path, ...] = (),
    qualified_native_wheels: dict[str, dict[str, Any]] | None = None,
    native_qualification_sha256: str | None = None,
    source_root: Path | None = None,
    qualification_manifest: Path | None = None,
    install_source_root: Path | None = None,
) -> dict[str, Any]:
    command_results: list[dict[str, Any]] = []
    qualified_native_dependencies: list[dict[str, str]] = []
    qualification_path = qualification_manifest or _native_qualification_manifest(source_root)
    source_identity = release_source_identity(source_root) if source_root is not None else None
    if source_root is not None and (
        not install_full or isolated_environment_root is None or install_source_root is None
        or release_source_identity(install_source_root) != source_identity
    ):
        raise ValueError("Public audits require an isolated installation of the exact exported source")
    qualification_options = {"source_root": source_root, "qualification_manifest": qualification_path}
    if install_full:
        current_qualification = _qualified_native_wheel_receipts(**qualification_options)
        current_qualification_sha256 = _sha256(qualification_path)
        if qualified_native_wheels is not None and qualified_native_wheels != current_qualification:
            raise ValueError("Native qualification receipts differ from the current candidate source")
        if native_qualification_sha256 is not None and native_qualification_sha256 != current_qualification_sha256:
            raise ValueError("Native qualification manifest changed before the disposable audit started")
        qualified_native_wheels = current_qualification
        native_qualification_sha256 = current_qualification_sha256
    package_environment = _package_environment()
    if isolated_environment_root is not None:
        # Preserve source/index sanitization and keep resolver downloads inside
        # the audit's owned disposable storage, including failed resolutions.
        package_environment["PIP_CACHE_DIR"] = str(isolated_environment_root / "pip-cache")
    audit_python = str(bootstrap_python)
    if isolated_environment_root is not None:
        create_environment = [bootstrap_python, "-m", "venv", str(isolated_environment_root)]
        created = _recorded_run(command_results, create_environment, env=package_environment)
        audit_python = str(_venv_python(isolated_environment_root))
        if created.returncode != 0:
            audit_python = str(bootstrap_python)

    install_report_path = output / "pip-install-report.json"
    build_install_report_path = output / "build-install-report.json"
    pip_check_path = output / "pip-check.txt"
    wheel_inventory_path = output / "wheel-inventory.json"
    sbom_path = output / "sbom.cdx.json"
    freeze_path = output / "python-freeze.txt"
    audit_path = output / "pip-audit.json"

    if install_full and all(row["return_code"] == 0 for row in command_results):
        commands = [
            [
                audit_python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX_URL,
                f"pip=={PIP_VERSION}",
            ],
            [
                audit_python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--index-url",
                PYPI_INDEX_URL,
                f"pip-audit=={PIP_AUDIT_VERSION}",
            ],
            _build_dependency_install_command(
                python_executable=audit_python,
                constraints=constraints,
                install_report_path=build_install_report_path,
                source_root=source_root,
            ),
            _full_install_command(
                python_executable=audit_python,
                constraints=constraints,
                install_report_path=install_report_path,
                torch_cpu_index=torch_cpu_index,
                native_wheels=native_wheels,
                source_root=install_source_root,
            ),
        ]
        for cmd in commands:
            proc = _recorded_run(command_results, cmd, env=package_environment)
            if proc.returncode != 0:
                break

        if all(row["return_code"] == 0 for row in command_results):
            pip_check = subprocess.run(
                [audit_python, "-m", "pip", "check"],
                cwd=ROOT,
                env=package_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            pip_check_path.write_text(pip_check.stdout + pip_check.stderr, encoding="utf-8")
            command_results.append(
                {
                    "command": [audit_python, "-m", "pip", "check"],
                    "return_code": int(pip_check.returncode),
                }
            )
            if pip_check.returncode == 0:
                dependency_evidence = [
                    audit_python,
                    "tools/generate_dependency_evidence.py",
                    "--install-report",
                    str(install_report_path),
                    "--constraints",
                    str(constraints),
                    "--additional-install-report",
                    str(build_install_report_path),
                    "--output",
                    str(wheel_inventory_path),
                ]
                _recorded_run(command_results, dependency_evidence, env=package_environment)
                if command_results[-1]["return_code"] == 0:
                    source_error = _validate_dependency_sources(
                        wheel_inventory_path,
                        torch_cpu_index=torch_cpu_index,
                        native_wheels=native_wheels,
                        qualified_native_wheels=qualified_native_wheels,
                        source_root=source_root,
                    )
                    command_results.append(
                        {
                            "command": ["validate-dependency-sources"],
                            "return_code": 0 if source_error is None else 1,
                            **({"error": source_error} if source_error is not None else {}),
                        }
                    )
                    if source_error is None and qualified_native_wheels is not None:
                        inventory = json.loads(wheel_inventory_path.read_text(encoding="utf-8"))
                        qualified_native_dependencies = [
                            {
                                "package": _normalized_distribution_name(row.get("name")),
                                "wheel": str(row.get("artifact", "")),
                                "sha256": str(row.get("sha256", "")),
                                "source_identity_sha256": qualified_native_wheels[
                                    _normalized_distribution_name(row.get("name"))
                                ]["source_identity_sha256"],
                            }
                            for row in list(dict(inventory).get("packages", []) or [])
                            if isinstance(row, dict)
                            and _normalized_distribution_name(row.get("name")) in qualified_native_wheels
                        ]

    if all(row["return_code"] == 0 for row in command_results):
        if isolated_environment_root is None:
            write_sbom(sbom_path)
        else:
            _recorded_run(
                command_results,
                [audit_python, "tools/generate_python_sbom.py", "--output", str(sbom_path)],
                env=package_environment,
            )

    if all(row["return_code"] == 0 for row in command_results):
        freeze = subprocess.run(
            [audit_python, "-m", "pip", "freeze", "--all"],
            cwd=ROOT,
            env=package_environment,
            text=True,
            capture_output=True,
            check=False,
        )
        freeze_path.write_text(freeze.stdout, encoding="utf-8")
        command_results.append(
            {
                "command": [audit_python, "-m", "pip", "freeze", "--all"],
                "return_code": int(freeze.returncode),
            }
        )
        if freeze.returncode == 0:
            audit_cmd = [
                audit_python,
                "-m",
                "pip_audit",
                "--format",
                "json",
                "--output",
                str(audit_path),
            ]
            _recorded_run(command_results, audit_cmd, env=package_environment)

    artifacts = []
    for path in (
        install_report_path,
        build_install_report_path,
        pip_check_path,
        wheel_inventory_path,
        sbom_path,
        freeze_path,
        audit_path,
    ):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if native_wheels:
        # Do not attest a wheelhouse rewritten while the disposable audit ran.
        _qualified_native_wheels(
            native_wheels[0].parent,
            qualified_receipts=qualified_native_wheels,
        )
    if qualified_native_wheels is not None:
        if (
            native_qualification_sha256 is None
            or _sha256(qualification_path) != native_qualification_sha256
            or _qualified_native_wheel_receipts(**qualification_options) != qualified_native_wheels
        ):
            raise ValueError("Native qualification changed while the disposable audit ran")
    if source_root is not None and release_source_identity(source_root) != source_identity:
        raise ValueError("Audited source export changed while the disposable audit ran")
    passed = bool(command_results) and all(row["return_code"] == 0 for row in command_results)
    manifest = {
        "schema_version": 1,
        "kind": "oel_supply_chain_gate",
        "product": "oel-public" if source_root is not None else "oel-pro",
        "package_version": _package_version(source_root),
        **({"source_identity": source_identity} if source_identity is not None else {}),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": audit_python,
        "bootstrap_python": bootstrap_python,
        "isolated_environment": isolated_environment_root is not None,
        "dependency_sources": {
            "primary_index": PYPI_INDEX_URL,
            "pytorch_cpu_index": PYTORCH_CPU_INDEX_URL if torch_cpu_index else None,
            "pip_environment_sanitized": True,
            "qualified_native_wheels": qualified_native_dependencies,
            "native_qualification_sha256": native_qualification_sha256,
        },
        "git": git_provenance(),
        "audit_exceptions": [],
        "commands": command_results,
        "artifacts": artifacts,
        "passed": passed,
    }
    manifest_path = output / "supply-chain-gate.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def run_supply_chain_gate(
    output_dir: str | Path,
    *,
    install_full: bool = False,
    python_executable: str = sys.executable,
    constraints_file: str | Path | None = None,
    torch_cpu_index: bool = False,
    native_wheelhouse: str | Path | None = None,
    source_root: str | Path | None = None,
    native_qualification_manifest: str | Path | None = None,
) -> dict[str, Any]:
    if native_wheelhouse is not None and not install_full:
        raise ValueError("Native wheel inputs require full-profile installation")
    public_root = Path(source_root).expanduser().resolve() if source_root is not None else None
    qualification_path = (
        Path(native_qualification_manifest).expanduser().resolve()
        if native_qualification_manifest is not None else _native_qualification_manifest(public_root)
    )
    if public_root is not None:
        if not install_full or native_wheelhouse is None or native_qualification_manifest is None:
            raise ValueError("A public source audit requires full installation, native wheels, and their qualification manifest")
        from tools.check_public_export import check_public_export

        errors = check_public_export(public_root)
        if errors:
            raise ValueError("Public audit source boundary check failed:\n- " + "\n- ".join(errors))
    native_qualification = (
        _qualified_native_wheel_receipts(source_root=public_root, qualification_manifest=qualification_path)
        if install_full else None
    )
    native_qualification_sha256 = (
        _sha256(qualification_path) if native_qualification is not None else None
    )
    native_wheels = (
        ()
        if native_wheelhouse is None
        else _qualified_native_wheels(
            Path(native_wheelhouse),
            qualified_receipts=native_qualification,
        )
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if constraints_file is None:
        constraints = (public_root or ROOT) / "constraints" / f"py{sys.version_info.major}{sys.version_info.minor}.txt"
    else:
        constraints = Path(constraints_file).expanduser().resolve()

    if install_full:
        if not constraints.is_file():
            raise FileNotFoundError(f"No approved constraints file for this interpreter: {constraints}")
        with tempfile.TemporaryDirectory(prefix="oel-supply-chain-") as temporary_root:
            install_source_root = None
            if public_root is not None:
                # pip may write build/egg-info into its input directory. Install
                # a byte-identical copy while keeping the attested export immutable.
                install_source_root = Path(temporary_root) / "public-source"
                for source in _source_files(public_root):
                    destination = install_source_root / source.relative_to(public_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                if release_source_identity(install_source_root) != release_source_identity(public_root):
                    raise ValueError("Public audit installation copy differs from the exported source")
            return _run_supply_chain_gate_in_environment(
                output,
                install_full=True,
                bootstrap_python=python_executable,
                constraints=constraints,
                torch_cpu_index=torch_cpu_index,
                isolated_environment_root=Path(temporary_root) / "audit-env",
                native_wheels=native_wheels,
                qualified_native_wheels=native_qualification,
                native_qualification_sha256=native_qualification_sha256,
                source_root=public_root,
                qualification_manifest=qualification_path,
                install_source_root=install_source_root,
            )
    return _run_supply_chain_gate_in_environment(
        output,
        install_full=False,
        bootstrap_python=python_executable,
        constraints=constraints,
        torch_cpu_index=torch_cpu_index,
        isolated_environment_root=None,
        qualified_native_wheels=native_qualification,
        native_qualification_sha256=native_qualification_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and audit OEL Python supply-chain release evidence.")
    parser.add_argument("--output-dir", default="outputs/supply_chain")
    parser.add_argument(
        "--install-full",
        action="store_true",
        help="Install pip-audit and OEL's full dependency profile before generating evidence.",
    )
    parser.add_argument(
        "--constraints",
        help="Approved constraints file (defaults to constraints/py<major><minor>.txt).",
    )
    parser.add_argument(
        "--torch-cpu-index",
        action="store_true",
        help=(
            "Resolve Torch from PyTorch's official CPU wheel index while resolving the full profile. "
            "Intended for disk-bounded Linux audit runners."
        ),
    )
    parser.add_argument("--native-wheelhouse", help="Local orbit/game wheels matching this release's committed host/interpreter qualification")
    parser.add_argument("--source-root", type=Path, help="Generated public source export to audit independently")
    parser.add_argument("--native-qualification-manifest", type=Path, help="Native qualification receipt bound to the audited source")
    args = parser.parse_args(argv)
    manifest = run_supply_chain_gate(
        args.output_dir,
        install_full=bool(args.install_full),
        constraints_file=args.constraints,
        torch_cpu_index=bool(args.torch_cpu_index),
        native_wheelhouse=args.native_wheelhouse,
        source_root=args.source_root,
        native_qualification_manifest=args.native_qualification_manifest,
    )
    print(f"Evidence manifest: {manifest['manifest']}")
    print(f"Supply-chain gate: {'PASS' if manifest['passed'] else 'FAIL'}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
