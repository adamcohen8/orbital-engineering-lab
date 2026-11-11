"""Small installer-authenticity regressions for the v0.36 follow-up audit.

Most cases use hand-built trees so the trust-boundary failures stay tiny and
deterministic.  The final runtime case also provisions two disposable venvs
from a local wheel through the host's existing pip, exercising pip's actual
``RECORD`` and shebang rewrites without a download or wheel cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from sim.installation import manager as installation_manager
from sim.installation import trusted_dispatcher, trusted_verifier
from sim.installation.contracts import (
    RELEASE_MANIFEST_SCHEMA,
    SCENARIO_SCHEMA,
    ContractError,
    canonical_json_bytes,
    release_manifest_digest,
    sha256_bytes,
    sha256_file,
)
from sim.installation.integrity import (
    RUNTIME_INVENTORY_SCHEMA,
    build_installation_integrity,
    build_runtime_inventory,
    build_source_inventory,
    validate_installation_integrity,
    verify_runtime_inventory,
    verify_source_inventory,
)
from sim.installation.manager import (
    _remove_bootstrap_runtime_packages,
    _remove_bytecode_caches,
    _remove_unbound_distribution_metadata,
    _remove_unbound_runtime_convenience,
    install_release,
    rotate_trusted_release_keys,
    verify_installation,
    write_launchers,
)
from sim.installation.paths import InstallationPaths
from sim.installation.signing import (
    RSAPublicKey,
    generate_rsa_private_key,
    public_key_to_json,
    sign_payload,
)
from sim.installation.trusted_verifier import _runtime_python


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
    return f"sha256={encoded}"


def _inventory_digest(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "digest"}
    return sha256_bytes(canonical_json_bytes(unsigned))


def _approved_external_provider() -> Path:
    """Return the actual interpreter base used by the trusted preflight gate."""

    return Path(getattr(sys, "_base_executable", sys.executable)).expanduser().resolve()


def _minimal_runtime_inventory(profile: str) -> dict[str, Any]:
    inventory: dict[str, Any] = {
        "schema_version": RUNTIME_INVENTORY_SCHEMA,
        "profile": profile,
        "wheels": [],
        "layout": {
            "interpreter": "bin/python",
            "provider": {"boundary": "external-interpreter", "trust": "operator-owned"},
            "extras": [],
        },
    }
    inventory["digest"] = _inventory_digest(inventory)
    return inventory


def _runtime_fixture(
    tmp_path: Path,
    *,
    script: tuple[bytes, bytes] | None = None,
    archive_record: bytes | None = None,
    installed_record: bytes | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    """Create a tiny venv-shaped tree plus a signed-runtime-shaped inventory.

    ``script`` is ``(archive_bytes, installed_bytes)``.  The provider is an
    external file so the fixture also exercises the managed interpreter
    boundary used by the verifier.
    """

    root = tmp_path / "runtime"
    site = root / "lib" / "python3.14" / "site-packages" / "demo-1.0.dist-info"
    scripts = root / "bin"
    site.mkdir(parents=True)
    scripts.mkdir(parents=True)
    provider = tmp_path / "operator-python"
    provider.write_bytes(b"operator python provider\n")
    (scripts / "python").symlink_to(provider)
    (root / "pyvenv.cfg").write_text(
        f"home = {provider.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    metadata = b"Name: demo\nVersion: 1.0\n"
    (site / "METADATA").write_bytes(metadata)
    files: list[dict[str, Any]] = [
        {
            "path": "demo-1.0.dist-info/METADATA",
            "install_path": "site-packages/demo-1.0.dist-info/METADATA",
            "bytes": len(metadata),
            "sha256": _digest(metadata),
            "executable": False,
        }
    ]
    record_lines = [
        f"demo-1.0.dist-info/METADATA,{_record_digest(metadata)},{len(metadata)}",
    ]
    if script is not None:
        archive_script, installed_script = script
        target = scripts / "demo-tool"
        target.write_bytes(installed_script)
        target.chmod(0o755)
        files.append(
            {
                "path": "demo-1.0.data/scripts/demo-tool",
                "install_path": "scripts/demo-tool",
                "bytes": len(archive_script),
                "sha256": _digest(archive_script),
                "installed_bytes": len(archive_script),
                "installed_sha256": _digest(archive_script),
                "pip_placeholder_shebang": archive_script.startswith(b"#!python"),
                "executable": True,
            }
        )
        record_lines.append(f"bin/demo-tool,{_record_digest(installed_script)},{len(installed_script)}")
    actual_record = installed_record or (
        ("\n".join(record_lines) + "\ndemo-1.0.dist-info/RECORD,,\n").encode()
    )
    (site / "RECORD").write_bytes(actual_record)
    record_member = archive_record if archive_record is not None else actual_record
    files.append(
        {
            "path": "demo-1.0.dist-info/RECORD",
            "install_path": "site-packages/demo-1.0.dist-info/RECORD",
            "bytes": len(record_member),
            "sha256": _digest(record_member),
            "executable": False,
        }
    )
    wheel = {
        "name": "demo-1.0-py3-none-any.whl",
        "bytes": 1,
        "sha256": "0" * 64,
        "files": files,
        "record": "demo-1.0.dist-info/RECORD",
        "generated_scripts": [],
    }
    inventory: dict[str, Any] = {
        "schema_version": RUNTIME_INVENTORY_SCHEMA,
        "profile": "core",
        "wheels": [wheel],
        "layout": {
            "interpreter": "bin/python",
            "provider": {"boundary": "external-interpreter", "trust": "operator-owned"},
            "extras": [
                {
                    "path": "pyvenv.cfg",
                    "kind": "venv-config",
                    "include_system_site_packages": False,
                    "provider_boundary": "external-interpreter",
                },
                {
                    "path": "bin/python",
                    "kind": "provider-alias",
                    "provider_boundary": "external-interpreter",
                },
            ],
        },
    }
    inventory["digest"] = _inventory_digest(inventory)
    return root, inventory, provider


def test_relocated_wheel_headers_remain_content_bound_and_inside_runtime(tmp_path: Path) -> None:
    from sim.installation.integrity import _wheel_install_path

    root, inventory, provider = _runtime_fixture(tmp_path)
    member = "demo-1.0.data/headers/demo.h"
    symbolic = _wheel_install_path(PurePosixPath(member))
    header = root / symbolic.removeprefix("runtime-data/")
    header.parent.mkdir(parents=True)
    data = b"/* authenticated header */\n"
    header.write_bytes(data)
    inventory["wheels"][0]["files"].append({
        "path": member, "install_path": symbolic,
        "bytes": len(data), "sha256": _digest(data), "executable": False,
    })
    inventory["digest"] = _inventory_digest(inventory)
    record = root / "lib/python3.14/site-packages/demo-1.0.dist-info/RECORD"
    relative = "../../../" + header.relative_to(root).as_posix()
    original = record.read_text(encoding="utf-8")
    record.write_text(original + f"{relative},{_record_digest(data)},{len(data)}\n", encoding="utf-8")
    assert not verify_runtime_inventory(root, inventory, interpreter=root / "bin/python", trusted_provider=provider)
    header.write_bytes(b"tampered\n")
    assert verify_runtime_inventory(root, inventory, interpreter=root / "bin/python", trusted_provider=provider)
    header.write_bytes(data)
    record.write_text(original + f"../{relative},{_record_digest(data)},{len(data)}\n", encoding="utf-8")
    assert verify_runtime_inventory(root, inventory, interpreter=root / "bin/python", trusted_provider=provider)


@pytest.mark.parametrize("trampoline", [False, True])
def test_authenticated_runtime_scripts_survive_transaction_rename(tmp_path: Path, trampoline: bool) -> None:
    from sim.installation.integrity import _rebind_runtime_scripts

    staged = tmp_path / "runtime"
    body = b"print('authenticated body')\n"
    archive = b"#!python\n" + body
    installed = b"#!" + os.fsencode(staged / "bin/python") + b"\n" + body
    if trampoline:
        installed = b"#!/bin/sh\n'''exec' '" + os.fsencode(staged / "bin/python") + b"' \"$0\" \"$@\"\n' '''\n" + body
    root, inventory, provider = _runtime_fixture(tmp_path, script=(archive, installed))
    assert not verify_runtime_inventory(root, inventory, interpreter=root / "bin/python", trusted_provider=provider)
    published = tmp_path / "published runtime with spaces"
    _rebind_runtime_scripts(root, inventory, published)
    root.rename(published)
    assert not verify_runtime_inventory(published, inventory, interpreter=published / "bin/python", trusted_provider=provider)
    (published / "bin/demo-tool").write_bytes(b"#!/bin/sh\necho tampered\n")
    assert verify_runtime_inventory(published, inventory, interpreter=published / "bin/python", trusted_provider=provider)


def test_installed_record_is_verified_semantically_not_as_archive_bytes(tmp_path: Path) -> None:
    archive_record = b"demo-1.0.dist-info/RECORD,sha256=archive-only,99\n"
    root, inventory, _provider = _runtime_fixture(tmp_path, archive_record=archive_record)

    issues = verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python")

    assert issues == []


def test_pip_rewritten_shebang_is_bound_to_the_external_provider(tmp_path: Path) -> None:
    provider = tmp_path / "operator-python"
    archive_script = b"#!python\nprint('demo')\n"
    installed_script = f"#!{provider}\nprint('demo')\n".encode()
    root, inventory, _provider = _runtime_fixture(
        tmp_path,
        script=(archive_script, installed_script),
    )

    issues = verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python")

    assert issues == []


def test_tampered_script_shebang_does_not_match_signed_provider(tmp_path: Path) -> None:
    other_provider = tmp_path / "other-python"
    other_provider.write_bytes(b"other provider\n")
    archive_script = b"#!python\nprint('demo')\n"
    installed_script = f"#!{other_provider}\nprint('demo')\n".encode()
    root, inventory, _provider = _runtime_fixture(
        tmp_path,
        script=(archive_script, installed_script),
    )

    issues = verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python")

    assert any("script mismatch" in issue for issue in issues)


def _write_tiny_runtime_wheel(path: Path) -> Path:
    """Write a valid, dependency-free wheel with data and script entry points."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dist_info = "tiny_oel_fixture-1.0.0.dist-info"
    members: dict[str, bytes] = {
        "tiny_demo/__init__.py": (
            b"def main():\n"
            b"    print('tiny runtime fixture')\n"
            b"    return 0\n"
        ),
        "tiny_demo/data.txt": b"ordinary wheel data\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\n"
            b"Name: tiny-oel-fixture\n"
            b"Version: 1.0.0\n"
            b"\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: oel-lane04\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
            b"\n"
        ),
        f"{dist_info}/entry_points.txt": b"[console_scripts]\ntiny-cli = tiny_demo:main\n",
        "tiny_oel_fixture-1.0.0.data/scripts/ordinary-tool": (
            b"#!python\n"
            b"from tiny_demo import main\n"
            b"raise SystemExit(main())\n"
        ),
    }
    record_rows = [
        f"{name},{_record_digest(data)},{len(data)}"
        for name, data in sorted(members.items())
    ]
    members[f"{dist_info}/RECORD"] = (
        "\n".join(record_rows) + f"\n{dist_info}/RECORD,,\n"
    ).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if name.endswith("/ordinary-tool") else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return path


def _runtime_python_path(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _prepare_real_pip_runtime(root: Path, wheel: Path) -> Path:
    """Install the tiny wheel through host pip's ``--python`` target mode."""

    root.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=False, clear=True, symlinks=os.name != "nt").create(root)
    python = _runtime_python_path(root)
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or "--python" not in (probe.stdout + probe.stderr):
        pytest.skip("the existing host pip lacks --python target installation support")
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("PIP_"):
            environment.pop(name, None)
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-index",
            "--no-deps",
            "--no-compile",
            "--no-cache-dir",
            str(wheel),
        ],
        cwd=wheel.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "host pip could not install the tiny offline wheel:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    # Keep this sequence aligned with manager._install_runtime.  These helpers
    # remove pip's provisioning-only metadata, bytecode and activation aliases
    # before the runtime is qualified and signed.
    _remove_bootstrap_runtime_packages(root)
    _remove_unbound_distribution_metadata(root)
    _remove_bytecode_caches(root)
    _remove_unbound_runtime_convenience(root)
    return python


def _prepare_structural_runtime(root: Path) -> Path:
    """Create the venv surface needed to bind generated wheel scripts.

    This deliberately avoids a second pip invocation in the profile-artifact
    negative case.  The runtime inventory builder only needs the installed
    entry-point wrapper and the authenticated venv/provider layout to produce
    the same schema that a qualified runtime would carry.
    """

    if os.name == "nt":
        pytest.skip("the structural provider-boundary fixture is POSIX-only")
    provider = _approved_external_provider()
    site_packages = root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    scripts = root / "bin"
    site_packages.mkdir(parents=True)
    scripts.mkdir(parents=True)
    for name in ("python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"):
        (scripts / name).symlink_to(provider)
    (root / "pyvenv.cfg").write_text(
        f"home = {provider.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    (root / "lib64").symlink_to("lib", target_is_directory=True)
    generated = scripts / "tiny-cli"
    generated.write_text(
        f"#!{provider}\n"
        "from tiny_demo import main\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    generated.chmod(0o755)
    return root


def test_real_pip_runtime_inventory_survives_recreation_and_detects_tamper(tmp_path: Path) -> None:
    """Qualify two real pip installs and reject body, shebang and RECORD edits."""

    if os.name == "nt":
        pytest.skip("this lane exercises the POSIX symlink-provider venv layout")
    wheel = _write_tiny_runtime_wheel(tmp_path / "wheelhouse" / "tiny_oel_fixture-1.0.0-py3-none-any.whl")
    runtime_a = tmp_path / "runtime with spaces-a"
    python_a = _prepare_real_pip_runtime(runtime_a, wheel)
    inventory = build_runtime_inventory(
        [wheel],
        interpreter=python_a.relative_to(runtime_a).as_posix(),
        provider={"boundary": "external-interpreter", "trust": "operator-owned"},
        profile="core",
        runtime_root=runtime_a,
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "source-marker.txt").write_text("qualified by the real wheel fixture\n", encoding="utf-8")
    integrity = build_installation_integrity(
        source,
        profile="core",
        runtime_profiles={"core": inventory},
    )
    assert validate_installation_integrity(integrity)["runtime"] == inventory
    selected = integrity["runtime_profiles"]["core"]
    assert verify_runtime_inventory(runtime_a, selected, interpreter=python_a) == []

    # Recreate at a different path instead of copying absolute shebangs.  The
    # signed normalized script identity and pyvenv.cfg provider binding must
    # survive the new location.
    runtime_b = tmp_path / "runtime with spaces-b"
    python_b = _prepare_real_pip_runtime(runtime_b, wheel)
    assert verify_runtime_inventory(runtime_b, selected, interpreter=python_b) == []

    ordinary = runtime_b / "bin" / "ordinary-tool"
    original_ordinary = ordinary.read_bytes()
    ordinary.write_bytes(original_ordinary + b"# tampered body\n")
    body_issues = verify_runtime_inventory(runtime_b, selected, interpreter=python_b)
    assert any("script mismatch" in issue or "file mismatch" in issue for issue in body_issues)
    ordinary.write_bytes(original_ordinary)
    assert verify_runtime_inventory(runtime_b, selected, interpreter=python_b) == []

    original_shebang = ordinary.read_bytes()
    ordinary.write_bytes(b"#!/bad/provider\n" + original_shebang.split(b"\n", 1)[1])
    shebang_issues = verify_runtime_inventory(runtime_b, selected, interpreter=python_b)
    assert any("script interpreter mismatch" in issue or "script mismatch" in issue for issue in shebang_issues)
    ordinary.write_bytes(original_shebang)
    assert verify_runtime_inventory(runtime_b, selected, interpreter=python_b) == []

    record_candidates = sorted(
        runtime_b.glob("lib/python*/site-packages/tiny_oel_fixture-1.0.0.dist-info/RECORD")
    )
    assert len(record_candidates) == 1
    record = record_candidates[0]
    original_record = record.read_bytes()
    record.write_bytes(original_record + b"unexpected.py,,\n")
    record_issues = verify_runtime_inventory(runtime_b, selected, interpreter=python_b)
    assert any("runtime RECORD" in issue for issue in record_issues)
    record.write_bytes(original_record)
    assert verify_runtime_inventory(runtime_b, selected, interpreter=python_b) == []


def test_source_forged_bytecode_is_not_covered_by_signed_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    inventory = build_source_inventory(source)
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-314.pyc").write_bytes(b"forged bytecode")

    issues = verify_source_inventory(source, inventory)

    assert any("bytecode" in issue for issue in issues)


def test_runtime_forged_bytecode_is_not_covered_by_signed_inventory(tmp_path: Path) -> None:
    root, inventory, _provider = _runtime_fixture(tmp_path)
    cache = root / "lib" / "python3.14" / "site-packages" / "__pycache__"
    cache.mkdir()
    (cache / "demo.cpython-314.pyc").write_bytes(b"forged bytecode")

    issues = verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python")

    assert any("bytecode" in issue for issue in issues)


def test_runtime_unicode_provider_alias_is_signed_and_checked(tmp_path: Path) -> None:
    root, inventory, provider = _runtime_fixture(tmp_path)
    alias = root / "bin" / "𝜋thon"
    alias.symlink_to(provider)
    layout = build_runtime_inventory([], runtime_root=root)["layout"]
    inventory["layout"]["extras"] = layout["extras"]
    inventory["digest"] = _inventory_digest(inventory)
    assert any(row["path"] == "bin/𝜋thon" for row in layout["extras"])
    assert verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python") == []

    replacement = tmp_path / "untrusted-provider"
    replacement.write_bytes(b"untrusted provider\n")
    alias.unlink()
    alias.symlink_to(replacement)
    issues = verify_runtime_inventory(root, inventory, interpreter=root / "bin" / "python")
    assert "managed runtime provider alias mismatch: bin/𝜋thon" in issues


def test_runtime_interpreter_provider_must_remain_external(tmp_path: Path) -> None:
    root, inventory, _provider = _runtime_fixture(tmp_path)
    internal_provider = root / "internal-provider-python"
    internal_provider.write_bytes(b"forged provider\n")
    interpreter = root / "bin" / "python"
    interpreter.unlink()
    interpreter.symlink_to(internal_provider)

    issues = verify_runtime_inventory(root, inventory, interpreter=interpreter)

    assert any("outside the managed runtime" in issue for issue in issues)


def test_runtime_interpreter_provider_must_match_trusted_external_provider(tmp_path: Path) -> None:
    root, inventory, _provider = _runtime_fixture(tmp_path)
    malicious_provider = tmp_path / "malicious-external-python"
    malicious_provider.write_bytes(b"forged external provider\n")
    interpreter = root / "bin" / "python"
    interpreter.unlink()
    interpreter.symlink_to(malicious_provider)

    issues = verify_runtime_inventory(
        root,
        inventory,
        interpreter=interpreter,
        trusted_provider=Path(sys.executable),
    )

    assert any("trusted external verifier provider" in issue for issue in issues)


def test_all_runtime_python_aliases_must_match_approved_external_provider(tmp_path: Path) -> None:
    """A forged outside provider cannot become trusted by editing every alias."""

    root, inventory, _fixture_provider = _runtime_fixture(tmp_path)
    approved = _approved_external_provider()
    scripts = root / "bin"
    aliases = ["python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"]
    extras = list(inventory["layout"]["extras"])
    for name in aliases:
        alias = scripts / name
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(approved)
        if not any(str(extra.get("path")) == f"bin/{name}" for extra in extras):
            extras.append(
                {
                    "path": f"bin/{name}",
                    "kind": "provider-alias",
                    "provider_boundary": "external-interpreter",
                }
            )
    inventory["layout"]["extras"] = extras
    (root / "pyvenv.cfg").write_text(
        f"home = {approved.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    inventory["digest"] = _inventory_digest(inventory)
    assert verify_runtime_inventory(
        root,
        inventory,
        interpreter=scripts / "python",
        trusted_provider=approved,
    ) == []

    malicious = tmp_path / "outside" / "malicious-python"
    malicious.parent.mkdir()
    malicious.write_text(
        "#!/usr/bin/env python3\n"
        "raise SystemExit('malicious provider must never execute')\n",
        encoding="utf-8",
    )
    malicious.chmod(0o755)
    for name in aliases:
        alias = scripts / name
        alias.unlink()
        alias.symlink_to(malicious)
    # Match the forged provider's parent as well.  The semantic layout checks
    # must still fail because the operator-approved provider identity differs.
    (root / "pyvenv.cfg").write_text(
        f"home = {malicious.parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    issues = verify_runtime_inventory(
        root,
        inventory,
        interpreter=scripts / "python",
        trusted_provider=approved,
    )

    assert any("trusted external verifier provider" in issue for issue in issues)


def test_trusted_verifier_rejects_external_python_when_runtime_is_not_created(tmp_path: Path) -> None:
    root = tmp_path / "version-root"
    provider = tmp_path / "operator-python"
    provider.write_bytes(b"operator provider\n")

    with pytest.raises(ContractError, match="authenticated managed runtime"):
        _runtime_python(root, {"created": False, "python": str(provider)})


def test_profile_runtime_inventories_remain_selected_and_bound(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    core = _minimal_runtime_inventory("core")
    game = _minimal_runtime_inventory("game")

    integrity = build_installation_integrity(
        source,
        profile="core",
        runtime_profiles={"core": core, "game": game},
    )

    assert integrity["runtime"] == core
    assert set(integrity["runtime_profiles"]) == {"core", "game"}
    assert validate_installation_integrity(integrity)["runtime"] == core

    forged = dict(integrity)
    # Keep the authenticated top-level selector at ``core`` while replacing
    # the selected runtime with the distinct ``game`` profile.  Recomputing
    # the outer digest models an attacker who can edit mutable bytes but not
    # the signed profile binding.
    forged["runtime"] = dict(game)
    forged["digest"] = _inventory_digest(forged)
    with pytest.raises(ContractError, match="selected profile"):
        validate_installation_integrity(forged)


def test_compressed_inventory_preserves_validation_and_standalone_bootstrap(tmp_path: Path) -> None:
    import runpy

    import jsonschema

    from sim.installation.integrity_transport import pack_installation_integrity
    from tools.build_installable_release import _render_installers

    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    integrity = build_installation_integrity(
        source, profile="core", runtime_profiles={"core": _minimal_runtime_inventory("core")},
    )
    packed = pack_installation_integrity(integrity)
    schema_path = Path(__file__).resolve().parents[2] / "sim/installation/schemas/release-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/installationIntegrity"}).validate(packed)
    assert validate_installation_integrity(packed) == integrity
    output = tmp_path / "rendered"
    _render_installers(output, public_keys=None, base_url=None, channel_url=None)
    bootstrap = runpy.run_path(str(output / "bootstrap_install.py"))
    assert bootstrap["unpack_installation_integrity"](packed) == integrity
    assert bootstrap["_validate_bootstrap_integrity"]({
        "profiles": ["core"], "supply_chain": {"installation_integrity": packed},
    })
    forged = dict(integrity)
    forged["digest"] = "0" * 64
    with pytest.raises(ContractError, match="inventory digest"):
        validate_installation_integrity(pack_installation_integrity(forged))
    corrupted = {**packed, "bytes": 1}
    assert not bootstrap["_validate_bootstrap_integrity"]({
        "profiles": ["core"], "supply_chain": {"installation_integrity": corrupted},
    })


@pytest.fixture
def lane04_signing_keys() -> tuple[Any, dict[str, RSAPublicKey]]:
    private = generate_rsa_private_key("lane04-installer", bits=2048)
    public = {private.key_id: RSAPublicKey(key_id=private.key_id, n=private.n)}
    return private, public


def _paths(tmp_path: Path, public: dict[str, RSAPublicKey]) -> InstallationPaths:
    paths = InstallationPaths(tmp_path / "managed-data", tmp_path / "managed-config")
    paths.ensure()
    paths.trusted_release_keys.write_text(
        json.dumps({"keys": [public_key_to_json(key) for key in public.values()]}, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def _write_closed_runtime_wheel(path: Path, *, version: str) -> Path:
    """Write a tiny wheel with a complete RECORD for official fixtures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dist_info = f"oel_fixture-{version}.dist-info"
    members: dict[str, bytes] = {
        "oel_fixture/__init__.py": b"fixture_value = 1\n",
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.1\n"
            b"Name: oel-fixture\n"
            + f"Version: {version}\n\n".encode()
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\n"
            b"Generator: oel-lane04\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n\n"
        ),
    }
    rows = [
        f"{name},{_record_digest(data)},{len(data)}"
        for name, data in sorted(members.items())
    ]
    members[f"{dist_info}/RECORD"] = ("\n".join(rows) + f"\n{dist_info}/RECORD,,\n").encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    return path


def _materialize_closed_runtime_wheels(runtime_root: Path, wheel_paths: list[Path]) -> None:
    site_packages = runtime_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    for wheel in wheel_paths:
        with zipfile.ZipFile(wheel) as archive:
            for info in archive.infolist():
                if info.filename.endswith("/"):
                    continue
                target = site_packages / Path(info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                mode = (info.external_attr >> 16) & 0o777
                target.chmod(mode or 0o644)


def _signed_release(root: Path, version: str, private: Any) -> Path:
    release = root / f"release-{version}"
    source = release / f"source-{version}"
    (source / "sim").mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        f"[project]\nname='orbital-engineering-lab'\nversion='{version}'\n",
        encoding="utf-8",
    )
    (source / "run_simulation.py").write_text("print('fixture')\n", encoding="utf-8")
    (source / "sim" / "__init__.py").write_text("", encoding="utf-8")
    archive = release / f"orbital-engineering-lab-{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname=f"orbital-engineering-lab-{version}")
    wheel = _write_closed_runtime_wheel(release / "wheelhouse" / f"oel_fixture-{version}-py3-none-any.whl", version=version)
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "product": "orbital-engagement-lab",
        "edition": "public",
        "version": version,
        "channel": "stable",
        "published_at": "2026-10-01T00:00:00Z",
        "artifacts": [
            {
                "name": archive.name,
                "kind": "source",
                "path": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            },
            {
                "name": wheel.name,
                "kind": "wheel",
                "path": f"wheelhouse/{wheel.name}",
                "bytes": wheel.stat().st_size,
                "sha256": sha256_file(wheel),
                "media_type": "application/zip",
            },
        ],
        "platforms": [platform.system()],
        "architecture": platform.machine(),
        "python": {"requires": ">=3.10,<3.15"},
        "profiles": ["core"],
        "contracts": {"workspace": "oel.workspace.v1", "scenario": SCENARIO_SCHEMA},
        "supply_chain": {
            "installation_integrity": build_installation_integrity(
                source,
                [wheel],
                interpreter="Scripts/python.exe" if os.name == "nt" else "bin/python",
                provider={"boundary": "external-interpreter", "trust": "operator-owned"},
            ),
            "offline_runtime_qualification": {"python": platform.python_version()},
        },
    }
    signed = sign_payload(manifest, private)
    manifest_path = release / "release-manifest.json"
    manifest_path.write_text(json.dumps(signed, sort_keys=True), encoding="utf-8")
    return manifest_path


def _install_minimal_official(
    manifest: Path,
    *,
    paths: InstallationPaths,
    public: dict[str, RSAPublicKey],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Install a signed release with a minimal authenticated runtime surface.

    The manager tests mutate an otherwise valid official record.  Using a
    small patched provisioning result keeps those tests focused on the record
    mutation while still exercising the official runtime and provenance gates;
    the separate real-pip test above covers the unpatched qualification path.
    """

    if os.name == "nt":
        pytest.skip("the minimal provider-boundary fixture is POSIX-only")

    def fake_install_runtime(source_root: Path, runtime_root: Path, **kwargs: Any) -> dict[str, Any]:
        del source_root
        provider = _approved_external_provider()
        python = runtime_root / "bin" / "python"
        (runtime_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages").mkdir(
            parents=True
        )
        python.parent.mkdir(parents=True, exist_ok=True)
        for name in ("python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"):
            (python.parent / name).symlink_to(provider)
        (runtime_root / "pyvenv.cfg").write_text(
            f"home = {provider.parent}\ninclude-system-site-packages = false\n",
            encoding="utf-8",
        )
        (runtime_root / "lib64").symlink_to("lib", target_is_directory=True)
        wheel_paths = [Path(item) for item in list(kwargs.get("signed_wheels") or [])]
        if not wheel_paths:
            raise AssertionError("official fixture provisioning requires a signed wheel closure")
        _materialize_closed_runtime_wheels(runtime_root, wheel_paths)
        return {
            "python": str(python),
            "install_command": ["lane04-minimal-runtime"],
            "pip_check": "lane04 fixture",
        }

    monkeypatch.setattr(installation_manager, "_install_runtime", fake_install_runtime)
    receipt = install_release(
        manifest,
        paths=paths,
        public_keys=public,
        create_runtime=True,
        offline_wheelhouse=manifest.parent / "wheelhouse",
    )
    assert receipt["installation"]["status"] == "official"
    return receipt


def _make_dispatcher_bundle(root: Path) -> tuple[Path, Path]:
    """Create only the structural bundle shell needed by dispatcher.main."""

    root.mkdir(parents=True, exist_ok=True)
    dispatcher_path = root / "dispatcher.py"
    dispatcher_path.write_text("# lane04 dispatcher seam\n", encoding="utf-8")
    bundle = root / "trusted-verifier"
    for directory in (bundle / "sim", bundle / "sim" / "installation"):
        directory.mkdir(parents=True, exist_ok=True)
    for relative in (
        "sim/__init__.py",
        "sim/schema_versions.py",
        "sim/installation/__init__.py",
        "sim/installation/contracts.py",
        "sim/installation/signing.py",
        "sim/installation/integrity.py",
        "sim/installation/integrity_transport.py",
        "sim/installation/trusted_verifier.py",
    ):
        (bundle / relative).write_text("# structural test placeholder\n", encoding="utf-8")
    return dispatcher_path, bundle


def test_trusted_dispatcher_refuses_source_or_runtime_tamper_before_execve(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The external preflight blocks tampered inputs without starting code."""

    # The in-process dispatcher prepends its verifier bundle before execve.
    # Restore the complete import path even if an assertion interrupts the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    private, public = lane04_signing_keys
    original_cwd = Path.cwd()
    monkeypatch.chdir(original_cwd)
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)
    verified = trusted_verifier.verify_official_installation(
        "1.0.0",
        data_root=paths.data_root,
        config_root=paths.config_root,
    )
    assert verified["status"] == "official"

    dispatcher_path, _bundle = _make_dispatcher_bundle(tmp_path / "dispatcher")
    monkeypatch.setattr(trusted_dispatcher, "__file__", str(dispatcher_path))
    exec_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(trusted_dispatcher.os, "execve", lambda *arguments: exec_calls.append(arguments))

    runtime_python = paths.version_root("1.0.0") / "runtime" / "bin" / "python"
    malicious_provider = tmp_path / "outside" / "malicious-python"
    malicious_provider.parent.mkdir()
    malicious_provider.write_text(
        "#!/usr/bin/env python3\n"
        "raise SystemExit('malicious runtime must never execute')\n",
        encoding="utf-8",
    )
    malicious_provider.chmod(0o755)
    runtime_python.unlink()
    runtime_python.symlink_to(malicious_provider)

    with pytest.raises(ContractError, match="trusted external verifier provider"):
        trusted_verifier.verify_official_installation(
            "1.0.0",
            data_root=paths.data_root,
            config_root=paths.config_root,
        )
    result = trusted_dispatcher.main(
        [
            "--data-root",
            str(paths.data_root),
            "--config-root",
            str(paths.config_root),
            "--version",
            "1.0.0",
        ]
    )
    monkeypatch.chdir(original_cwd)
    assert result == 78
    assert exec_calls == []

    approved = _approved_external_provider()
    runtime_python.unlink()
    runtime_python.symlink_to(approved)
    source_root = next(paths.version_root("1.0.0").glob("source/*"))
    source_file = source_root / "sim" / "__init__.py"
    original_source = source_file.read_bytes()
    source_file.write_bytes(original_source + b"\n# forged source\n")
    with pytest.raises(ContractError, match="source inventory"):
        trusted_verifier.verify_official_installation(
            "1.0.0",
            data_root=paths.data_root,
            config_root=paths.config_root,
        )
    result = trusted_dispatcher.main(
        [
            "--data-root",
            str(paths.data_root),
            "--config-root",
            str(paths.config_root),
            "--version",
            "1.0.0",
        ]
    )
    monkeypatch.chdir(original_cwd)
    assert result == 78
    assert exec_calls == []
    source_file.write_bytes(original_source)


def test_external_trusted_dispatcher_blocks_tampered_inputs_before_managed_exec(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the generated launcher gate in a fresh isolated process."""

    if os.name == "nt":
        pytest.skip("the subprocess fixture exercises the POSIX launcher")

    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)
    launchers = write_launchers("1.0.0", paths=paths)
    dispatcher = Path(launchers["trusted_dispatcher"])
    provider = _approved_external_provider()
    command = [
        str(provider),
        "-I",
        "-B",
        str(dispatcher),
        "--data-root",
        str(paths.data_root),
        "--config-root",
        str(paths.config_root),
        "--version",
        "1.0.0",
    ]
    source_root = next(paths.version_root("1.0.0").glob("source/*"))
    source_init = source_root / "sim" / "__init__.py"
    original_source = source_init.read_bytes()
    marker = tmp_path / "managed-source-executed"
    source_init.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    finally:
        source_init.write_bytes(original_source)
    assert completed.returncode == 78, completed.stderr
    assert not marker.exists()

    malicious_marker = tmp_path / "malicious-provider-executed"
    malicious_provider = tmp_path / "outside" / "malicious-python"
    malicious_provider.parent.mkdir()
    malicious_provider.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        f"Path({str(malicious_marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    malicious_provider.chmod(0o755)
    runtime_python = paths.version_root("1.0.0") / "runtime" / "bin" / "python"
    runtime_python.unlink()
    runtime_python.symlink_to(malicious_provider)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    assert completed.returncode == 78, completed.stderr
    assert not malicious_marker.exists()


def _installed_record(root: Path) -> dict[str, Any]:
    return json.loads((root / "installation-record.json").read_text(encoding="utf-8"))


def test_manager_rejects_signed_manifest_from_another_version_root(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest_a = _signed_release(tmp_path, "1.0.0", private)
    manifest_b = _signed_release(tmp_path, "2.0.0", private)
    _install_minimal_official(manifest_a, paths=paths, public=public, monkeypatch=monkeypatch)

    root = paths.version_root("1.0.0")
    signed_b = json.loads(manifest_b.read_text(encoding="utf-8"))
    (root / "release-manifest.json").write_text(json.dumps(signed_b, sort_keys=True), encoding="utf-8")
    record = _installed_record(root)
    record["release_manifest_sha256"] = release_manifest_digest(signed_b)
    integrity = signed_b["supply_chain"]["installation_integrity"]
    record["provenance"]["installation_integrity_sha256"] = integrity["digest"]
    record["provenance"]["source_inventory_sha256"] = integrity["source"]["digest"]
    record["provenance"]["runtime_inventory_sha256"] = integrity["runtime"]["digest"]
    (root / "installation-record.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    result = verify_installation("1.0.0", paths=paths, full=False)

    assert result["status"] != "official"


def test_manager_rejects_profile_wheel_absent_from_signed_artifacts(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected signed wheel must also be a declared release artifact."""

    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest_path = _signed_release(tmp_path, "1.0.0", private)
    wheel = _write_tiny_runtime_wheel(manifest_path.parent / "tiny_oel_fixture-1.0.0-py3-none-any.whl")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("signature", None)
    runtime_root = _prepare_structural_runtime(tmp_path / "signed-runtime")
    selected_runtime = build_runtime_inventory(
        [wheel],
        interpreter="bin/python",
        provider={"boundary": "external-interpreter", "trust": "operator-owned"},
        profile="core",
        runtime_root=runtime_root,
    )
    integrity = manifest["supply_chain"]["installation_integrity"]
    integrity["runtime"] = selected_runtime
    integrity["runtime_profiles"] = {"core": selected_runtime}
    integrity["digest"] = _inventory_digest(integrity)
    manifest_path.write_text(json.dumps(sign_payload(manifest, private), sort_keys=True), encoding="utf-8")

    install_calls: list[tuple[Any, ...]] = []

    def fail_if_provisioned(*args: Any, **_kwargs: Any) -> dict[str, Any]:
        install_calls.append(args)
        raise AssertionError("runtime provisioning must follow signed artifact binding")

    monkeypatch.setattr(installation_manager, "_install_runtime", fail_if_provisioned)
    with pytest.raises(ContractError, match="wheel"):
        install_release(
            manifest_path,
            paths=paths,
            public_keys=public,
            create_runtime=True,
            offline_wheelhouse=manifest_path.parent / "wheelhouse",
        )
    assert install_calls == []


def test_manager_does_not_treat_mutable_developer_status_as_authenticated(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)
    root = paths.version_root("1.0.0")
    record = _installed_record(root)
    record["status"] = "developer"
    (root / "installation-record.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    result = verify_installation("1.0.0", paths=paths, full=False)

    assert result["status"] != "developer"


def test_manager_preserves_untrusted_disposition_for_legacy_installations(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)
    root = paths.version_root("1.0.0")
    record = _installed_record(root)
    record["status"] = "untrusted"
    (root / "installation-record.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    result = verify_installation("1.0.0", paths=paths, full=False)

    assert result["status"] == "untrusted"


def test_manager_binds_provenance_to_the_external_trust_anchor(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)
    root = paths.version_root("1.0.0")
    record = _installed_record(root)
    record["provenance"]["trust_anchor_sha256"] = "0" * 64
    (root / "installation-record.json").write_text(json.dumps(record, sort_keys=True), encoding="utf-8")

    result = verify_installation("1.0.0", paths=paths, full=False)

    assert result["status"] == "official"
    assert "diagnostics" in result
    assert any("anchor digest" in item for item in result["diagnostics"])


def test_nonrevoking_key_rotation_preserves_official_preflight_and_revocation_fails(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing installs survive an authorized rotation until their signer is revoked."""

    # execve is mocked below, so its preparatory sys.path mutation must be local.
    monkeypatch.setattr(sys, "path", list(sys.path))
    private, public = lane04_signing_keys
    original_cwd = Path.cwd()
    monkeypatch.chdir(original_cwd)
    old_public = next(iter(public.values()))
    paths = _paths(tmp_path, public)
    manifest = _signed_release(tmp_path, "1.0.0", private)
    _install_minimal_official(manifest, paths=paths, public=public, monkeypatch=monkeypatch)

    successor = generate_rsa_private_key("successor", bits=2048)
    successor_public = RSAPublicKey(key_id=successor.key_id, n=successor.n)
    retained_registry = sign_payload(
        {
            "schema_version": "oel.trusted-key-registry.v1",
            "published_at": "2026-10-02T00:00:00Z",
            "keys": [public_key_to_json(old_public), public_key_to_json(successor_public)],
        },
        private,
    )
    retained_path = tmp_path / "retained-keys.json"
    retained_path.write_text(json.dumps(retained_registry, sort_keys=True), encoding="utf-8")

    rotation = rotate_trusted_release_keys(retained_path, paths=paths, current_keys=public)
    assert rotation["key_ids"] == sorted((old_public.key_id, successor.key_id))
    verified = verify_installation("1.0.0", paths=paths, full=True)
    assert verified["status"] == "official"
    external = trusted_verifier.verify_official_installation(
        "1.0.0",
        data_root=paths.data_root,
        config_root=paths.config_root,
    )
    assert external["status"] == "official"

    launchers = write_launchers("1.0.0", paths=paths)
    monkeypatch.setattr(trusted_dispatcher, "__file__", launchers["trusted_dispatcher"])
    exec_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(trusted_dispatcher.os, "execve", lambda *arguments: exec_calls.append(arguments))
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    (caller_cwd / "relative.yaml").write_text("scenario: relative\n", encoding="utf-8")
    monkeypatch.chdir(caller_cwd)
    dispatch_arguments = [
        "--data-root",
        str(paths.data_root),
        "--config-root",
        str(paths.config_root),
        "--version",
        "1.0.0",
        "--",
        "sim",
        "--config",
        "relative.yaml",
    ]
    assert trusted_dispatcher.main(dispatch_arguments) == 0
    assert exec_calls
    assert Path.cwd() == caller_cwd
    command = exec_calls[0][1]
    environment = exec_calls[0][2]
    assert command[1:4] == ["-I", "-B", "-c"]
    assert "runpy.run_module('sim.installation.cli',run_name='__main__')" in command[4]
    record = _installed_record(paths.version_root("1.0.0"))
    installed_source = paths.version_root("1.0.0") / record["source"]["path"]
    assert command[5] == str(installed_source)
    assert command[-3:] == ["sim", "--config", "relative.yaml"]
    assert "PYTHONPATH" not in environment
    monkeypatch.chdir(original_cwd)

    revoked_old = RSAPublicKey(
        key_id=old_public.key_id,
        n=old_public.n,
        e=old_public.e,
        alg=old_public.alg,
        not_before=old_public.not_before,
        expires_at=old_public.expires_at,
        revoked=True,
    )
    revoked_registry = sign_payload(
        {
            "schema_version": "oel.trusted-key-registry.v1",
            "published_at": "2026-10-03T00:00:00Z",
            "keys": [public_key_to_json(revoked_old), public_key_to_json(successor_public)],
        },
        successor,
    )
    revoked_path = tmp_path / "revoked-keys.json"
    revoked_path.write_text(json.dumps(revoked_registry, sort_keys=True), encoding="utf-8")
    current_keys = {old_public.key_id: old_public, successor_public.key_id: successor_public}
    rotation = rotate_trusted_release_keys(revoked_path, paths=paths, current_keys=current_keys)
    assert rotation["key_ids"] == sorted((old_public.key_id, successor.key_id))

    refused = verify_installation("1.0.0", paths=paths, full=True)
    assert refused["status"] != "official"
    with pytest.raises(ContractError, match="manifest.*trusted-key|trusted-key.*manifest|signature"):
        trusted_verifier.verify_official_installation(
            "1.0.0",
            data_root=paths.data_root,
            config_root=paths.config_root,
        )
    exec_calls.clear()
    assert trusted_dispatcher.main(dispatch_arguments) == 78
    monkeypatch.chdir(original_cwd)
    assert exec_calls == []


def test_key_rotation_rejects_noncanonical_older_timestamp(
    tmp_path: Path,
    lane04_signing_keys: tuple[Any, dict[str, RSAPublicKey]],
) -> None:
    private, public = lane04_signing_keys
    paths = _paths(tmp_path, public)
    paths.trusted_release_keys.write_text(
        json.dumps(
            {
                "published_at": "2026-10-01T00:00:00Z",
                "keys": [public_key_to_json(next(iter(public.values())))],
            }
        ),
        encoding="utf-8",
    )
    registry = sign_payload(
        {
            "schema_version": "oel.trusted-key-registry.v1",
            "published_at": "2026-9-30T00:00:00Z",
            "keys": [public_key_to_json(next(iter(public.values())))],
        },
        private,
    )
    registry_path = tmp_path / "older-keys.json"
    registry_path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="timestamp|published|rollback|ISO"):
        rotate_trusted_release_keys(registry_path, paths=paths, current_keys=public)
