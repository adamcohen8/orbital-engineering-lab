from __future__ import annotations

import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from sim.installation.archive import UnsafeArchiveError, safe_extract
from sim.installation.cli import _build_parser, _dispatch, _engine_lease, _source_from_record, _split_dispatch_argv
from sim.installation.contracts import (
    CHANNEL_INDEX_SCHEMA,
    RELEASE_MANIFEST_SCHEMA,
    ContractError,
    sha256_file,
    validate_release_manifest,
    version_satisfies,
)
from sim.installation.manager import (
    _source_and_python,
    _validate_offline_runtime_compatibility,
    activate,
    check_channel,
    configure_channel,
    configured_channel_url,
    download_release,
    download_release_wheelhouse,
    install_bundle,
    install_latest_release,
    install_release,
    pro_installation_available,
    rollback,
    rotate_trusted_release_keys,
    uninstall,
    verify_installation,
    write_launchers,
    write_support_receipt,
)
from sim.installation.paths import InstallationPaths
from sim.installation.signing import (
    RSAPublicKey,
    generate_rsa_private_key,
    private_key_to_json,
    public_key_to_json,
    sign_payload,
    verify_payload,
)
from sim.installation.state import StateLock, atomic_write_json, read_state
from sim.installation.workspace import (
    apply_migration,
    audit_workspace,
    init_workspace,
    load_workspace,
    plan_migration,
    plan_template_sync,
    register_workspace,
)
from sim.schema_versions import SCENARIO_SCHEMA_VERSION
from tools.build_installable_release import (
    _collect_wheelhouse,
    _official_signing_material,
    _render_installers,
    _source_files,
    _verify_offline_runtime_install,
    build_release,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def signing_keys() -> tuple[object, dict[str, RSAPublicKey]]:
    private = generate_rsa_private_key("installation-tests", bits=2048)
    return private, {private.key_id: RSAPublicKey(key_id=private.key_id, n=private.n)}


def _paths(tmp_path: Path) -> InstallationPaths:
    return InstallationPaths(tmp_path / "managed data", tmp_path / "managed config")


@pytest.mark.parametrize(
    "script",
    [
        "tools/build_installable_release.py",
        "tools/generate_release_signing_key.py",
        "tools/generate_license_signing_key.py",
    ],
)
def test_release_tools_support_direct_invocation(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _release(
    root: Path,
    version: str,
    private_key: object,
    *,
    edition: str = "public",
    artifact_platform: str | None = None,
) -> tuple[Path, Path]:
    release = root / f"release-{version}"
    source = release / f"source-{version}"
    source.mkdir(parents=True)
    (source / "pyproject.toml").write_text(
        f'[project]\nname = "orbital-engineering-lab"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (source / "run_simulation.py").write_text("print('fixture')\n", encoding="utf-8")
    (source / "sim").mkdir()
    (source / "sim" / "__init__.py").write_text("", encoding="utf-8")
    archive = release / f"orbital-engineering-lab-{version}-{edition}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname=f"orbital-engineering-lab-{version}")
    artifact = {
        "name": archive.name,
        "kind": "source",
        "path": archive.name,
        "url": archive.as_uri(),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        **({"platform": artifact_platform} if artifact_platform else {}),
    }
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "product": "orbital-engagement-lab",
        "edition": edition,
        "version": version,
        "channel": "stable",
        "published_at": "2026-08-15T00:00:00Z",
        "artifacts": [artifact],
        "platforms": [platform.system()],
        "python": {"requires": ">=3.10,<3.15"},
        "profiles": ["core"],
        "contracts": {"workspace": "oel.workspace.v1", "scenario": SCENARIO_SCHEMA_VERSION},
    }
    signed = sign_payload(manifest, private_key)  # type: ignore[arg-type]
    manifest_path = release / "release-manifest.json"
    manifest_path.write_text(json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, archive


def test_version_range_grammar_is_bounded() -> None:
    assert version_satisfies("0.25.1", ">=0.24,<0.27")
    assert not version_satisfies("0.27.0", ">=0.24,<0.27")
    assert not version_satisfies("0.25.0", "~=0.25")


def test_launcher_preserves_downstream_flags() -> None:
    head, downstream = _split_dispatch_argv(
        ["--workspace", "workspace with spaces", "sim", "--quickstart", "--validate-only"]
    )
    assert head == ["--workspace", "workspace with spaces", "sim"]
    assert downstream == ["--quickstart", "--validate-only"]


def test_launcher_advertises_fswdk_only_when_edition_contains_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sim.installation.cli._fswdk_available", lambda: False)
    assert "fswdk" not in _build_parser().format_help()
    head, downstream = _split_dispatch_argv(["fswdk", "--help"])
    assert head == ["fswdk", "--help"]
    assert downstream is None

    monkeypatch.setattr("sim.installation.cli._fswdk_available", lambda: True)
    assert "fswdk" in _build_parser().format_help()
    head, downstream = _split_dispatch_argv(["fswdk", "--help"])
    assert head == ["fswdk"]
    assert downstream == ["--help"]


def test_launcher_advertises_pro_installation_only_when_license_verifier_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sim.installation.cli.pro_installation_available", lambda: False)
    public_only = _build_parser()
    parsed = public_only.parse_args(["update", "check", "--edition", "public"])
    assert parsed.edition == "public"
    with pytest.raises(SystemExit):
        public_only.parse_args(["update", "check", "--edition", "pro"])

    monkeypatch.setattr("sim.installation.cli.pro_installation_available", lambda: True)
    private = _build_parser().parse_args(["update", "check", "--edition", "pro"])
    assert private.edition == "pro"


def test_pro_installation_fails_closed_without_license_module(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, public = signing_keys
    manifest, _ = _release(tmp_path, "0.26.0", private, edition="pro")
    monkeypatch.setattr("sim.installation.manager.importlib.util.find_spec", lambda _name: None)
    assert pro_installation_available() is False
    with pytest.raises(ContractError, match="Managed OEL Pro installation is unavailable"):
        install_release(manifest, paths=_paths(tmp_path), public_keys=public, create_runtime=False)


def test_pro_installation_rejects_license_module_from_another_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_module = tmp_path / "other-oel" / "sim" / "licensing" / "offline.py"
    spec = importlib.util.spec_from_file_location("sim.licensing.offline", external_module)
    monkeypatch.setattr("sim.installation.manager.importlib.util.find_spec", lambda _name: spec)

    assert pro_installation_available() is False


def test_release_contract_rejects_unknown_fields(signing_keys: tuple[object, dict[str, RSAPublicKey]]) -> None:
    private, _ = signing_keys
    payload = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "product": "orbital-engagement-lab",
        "edition": "public",
        "version": "0.25.0",
        "channel": "stable",
        "published_at": "2026-08-15T00:00:00Z",
        "artifacts": [{"name": "x", "kind": "source", "path": "x", "bytes": 0, "sha256": "0" * 64}],
        "platforms": [platform.system()],
        "python": {},
        "profiles": ["core"],
        "contracts": {},
        "surprise": True,
    }
    with pytest.raises(ContractError, match="unsupported field"):
        validate_release_manifest(sign_payload(payload, private))  # type: ignore[arg-type]


@pytest.mark.parametrize("product", ["orbital-engagement-lab", "orbital-engineering-lab"])
def test_release_contract_accepts_both_product_identities(product: str) -> None:
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "product": product,
        "edition": "public",
        "version": "0.28.0",
        "channel": "stable",
        "published_at": "2026-08-15T00:00:00Z",
        "artifacts": [{"name": "source.tar.gz", "kind": "source", "path": "source.tar.gz", "bytes": 0, "sha256": "0" * 64}],
        "platforms": [platform.system()],
        "python": {"requires": ">=3.10,<3.15"},
        "profiles": ["core"],
        "contracts": {"workspace": "oel.workspace.v1", "scenario": SCENARIO_SCHEMA_VERSION},
    }
    assert validate_release_manifest(manifest)["product"] == product


def test_signature_tamper_expiry_and_revocation(signing_keys: tuple[object, dict[str, RSAPublicKey]]) -> None:
    private, keys = signing_keys
    signed = sign_payload({"value": 1}, private)  # type: ignore[arg-type]
    assert verify_payload(signed, keys)
    assert not verify_payload({**signed, "value": 2}, keys)
    now = datetime.now(timezone.utc)
    expired = RSAPublicKey(
        key_id=private.key_id,  # type: ignore[attr-defined]
        n=private.n,  # type: ignore[attr-defined]
        expires_at=now - timedelta(seconds=1),
    )
    assert not verify_payload(signed, {expired.key_id: expired}, now=now)
    revoked = RSAPublicKey(key_id=private.key_id, n=private.n, revoked=True)  # type: ignore[attr-defined]
    assert not verify_payload(signed, {revoked.key_id: revoked})


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "C:/drive"])
def test_safe_extract_rejects_path_escape(tmp_path: Path, member_name: str) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo(member_name)
        info.size = 1
        bundle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "output")


def test_safe_extract_rejects_duplicate_and_link(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.tar"
    with tarfile.open(duplicate, "w") as bundle:
        for value in (b"a", b"b"):
            info = tarfile.TarInfo("same")
            info.size = 1
            bundle.addfile(info, io.BytesIO(value))
    with pytest.raises(UnsafeArchiveError, match="duplicate"):
        safe_extract(duplicate, tmp_path / "duplicate-output")
    linked = tmp_path / "link.tar"
    with tarfile.open(linked, "w") as bundle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        bundle.addfile(info)
    with pytest.raises(UnsafeArchiveError, match="special"):
        safe_extract(linked, tmp_path / "link-output")


def test_state_lock_is_exclusive_and_atomic(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    atomic_write_json(state, {"value": 1})
    assert read_state(state) == {"value": 1}
    lock = tmp_path / "lock"
    with StateLock(lock, operation="first"):
        with pytest.raises(RuntimeError, match="Another OEL state transaction"):
            with StateLock(lock, operation="second"):
                pass
    assert not lock.exists()


def test_workspace_audit_preserves_user_source_and_does_not_import_candidates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "user workspace"
    init_workspace(workspace_root, engine_version="0.25.0")
    config = workspace_root / "configs" / "legacy.yaml"
    config.write_text("scenario_name: user_owned\n", encoding="utf-8")
    candidate = workspace_root / "fsw" / "candidate.yaml"
    candidate.write_text(
        yaml.safe_dump(
            {
                "schema_version": "oel.fswdk.candidate.v1",
                "interfaces": {"onboard_contract": "oel.fsw.boundary.v2"},
                "module": "would_create_a_side_effect_if_imported",
            }
        ),
        encoding="utf-8",
    )
    before = {path: path.read_bytes() for path in (config, candidate)}

    report = audit_workspace(workspace_root, target_version="0.26.0")

    assert report["status"] == "migration_available"
    assert report["effects"] == {
        "user_source_modified": False,
        "workspace_metadata_written": True,
        "code_executed": False,
        "network_used": False,
    }
    assert {path: path.read_bytes() for path in before} == before


def test_workspace_manifest_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "target"
    init_workspace(target, engine_version="0.25.0")
    alias = tmp_path / "alias" / "oel-workspace.yaml"
    alias.parent.mkdir()
    alias.symlink_to(target / "oel-workspace.yaml")

    with pytest.raises(ContractError, match="manifest may not be a symbolic link"):
        load_workspace(alias)


def test_migration_is_explicit_recoverable_and_idempotent(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    init_workspace(workspace_root, engine_version="0.25.0")
    config = workspace_root / "configs" / "legacy.yaml"
    original = b"scenario_name: user_owned\n"
    config.write_bytes(original)

    plan = plan_migration(workspace_root, target_version="0.26.0")
    assert config.read_bytes() == original
    assert plan["status"] == "ready"
    receipt = apply_migration(plan["plan_path"])

    assert yaml.safe_load(config.read_text(encoding="utf-8"))["schema_version"] == SCENARIO_SCHEMA_VERSION
    backup = Path(next(item["backup"] for item in receipt["applied"] if item["path"] == "configs/legacy.yaml"))
    assert backup.read_bytes() == original
    repeated = apply_migration(plan["plan_path"])
    assert repeated["idempotent"] is True
    assert load_workspace(workspace_root)["engine"]["locked_version"] == "0.26.0"


def test_side_by_side_install_activation_rollback_integrity_and_uninstall(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    paths = _paths(tmp_path)
    manifest_a, _ = _release(tmp_path, "0.25.0", private)
    manifest_b, _ = _release(tmp_path, "0.26.0", private)
    workspace_root = tmp_path / "workspace"
    init_workspace(workspace_root, engine_version="0.25.0")
    register_workspace(workspace_root, registry_path=paths.workspaces_state)

    first = install_release(manifest_a, paths=paths, public_keys=keys, create_runtime=False)
    second = install_release(manifest_b, paths=paths, public_keys=keys, create_runtime=False)
    assert first["installation"]["status"] == "official"
    assert second["installation"]["status"] == "official"
    activate("0.25.0", paths=paths)
    activate("0.26.0", paths=paths)
    assert load_workspace(workspace_root)["engine"]["locked_version"] == "0.25.0"
    assert rollback(paths=paths)["current"] == "0.25.0"

    blocked = uninstall("0.25.0", paths=paths, dry_run=True)
    assert blocked["status"] == "blocked"
    assert blocked["blockers"]["workspaces"]
    source = next(paths.version_root("0.26.0").glob("source/*"))
    (source / "run_simulation.py").write_text("modified\n", encoding="utf-8")
    assert verify_installation("0.26.0", paths=paths, full=True)["status"] == "modified"


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtualenv interpreters are symlinks")
def test_managed_runtime_selection_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    root = paths.version_root("0.26.0")
    source = root / "source" / "orbital-engineering-lab-0.26.0"
    source.mkdir(parents=True)
    runtime_python = root / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(Path(sys.executable).resolve())
    atomic_write_json(
        root / "installation-record.json",
        {
            "runtime": {
                "created": True,
                "python": str(root / ".transaction" / "runtime" / "bin" / "python"),
            },
            "source": {"path": str(source.relative_to(root))},
        },
    )

    selected_source, selected_python = _source_and_python("0.26.0", paths)

    assert selected_source == source.resolve()
    assert selected_python == runtime_python.absolute()
    assert selected_python != runtime_python.resolve()

    dispatched_source, dispatched_python = _source_from_record("0.26.0", paths)
    assert dispatched_source == source.resolve()
    assert dispatched_python == runtime_python.absolute()
    assert dispatched_python != runtime_python.resolve()

    launchers = write_launchers("0.26.0", paths=paths)
    posix_launcher = Path(launchers["posix"]).read_text(encoding="utf-8")
    windows_launcher = Path(launchers["windows"]).read_text(encoding="utf-8")
    assert f"--data-root '{paths.data_root}'" in posix_launcher
    assert f"--config-root '{paths.config_root}'" in posix_launcher
    assert f'--data-root "{paths.data_root}"' in windows_launcher
    assert f'--config-root "{paths.config_root}"' in windows_launcher
    runtime_python.unlink()


def test_managed_dispatch_disables_source_bytecode_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    version_root = paths.version_root("0.26.0")
    version_root.mkdir(parents=True)
    atomic_write_json(
        version_root / "installation-record.json",
        {
            "edition": "public",
            "release_manifest_sha256": "fixture-manifest",
            "transaction_id": "fixture-transaction",
        },
    )
    source = tmp_path / "installed source"
    source.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sim.installation.cli._selected_engine",
        lambda _paths, _workspace: ("0.26.0", source, Path(sys.executable), "official"),
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("sim.installation.cli.subprocess.run", fake_run)

    assert _dispatch("doctor", [], paths=paths, workspace_path=None) == 0
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_managed_dispatch_routes_hosted_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    version_root = paths.version_root("0.26.0")
    version_root.mkdir(parents=True)
    atomic_write_json(
        version_root / "installation-record.json",
        {
            "edition": "public",
            "release_manifest_sha256": "fixture-manifest",
            "transaction_id": "fixture-transaction",
        },
    )
    source = tmp_path / "installed source"
    source.mkdir()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sim.installation.cli._selected_engine",
        lambda _paths, _workspace: ("0.26.0", source, Path(sys.executable), "official"),
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("sim.installation.cli.subprocess.run", fake_run)

    assert _dispatch("hosted", ["--help"], paths=paths, workspace_path=None) == 0
    assert captured["command"] == [sys.executable, "-m", "sim.hosted_client", "--help"]


@pytest.mark.parametrize(
    ("arguments", "module"),
    [
        (["capabilities", "--scope", "discovery"], "sim.study_planning"),
        (["schema", "oel.hosted_study_plan.v2"], "sim.study_planning"),
        (["preflight", "--help"], "sim.study_planning"),
        (["plan-review", "--help"], "sim.study_planning"),
        (["prepare-feedback", "--help"], "sim.study_planning"),
        (["build", "--help"], "sim.study"),
        (["inspect", "bundle"], "sim.study"),
        (["replay", "bundle"], "sim.study"),
        (["--help"], "sim.study"),
    ],
)
def test_dispatch_preserves_both_study_cli_workflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str], module: str
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        "sim.installation.cli._selected_engine",
        lambda _paths, _workspace: (None, tmp_path, Path(sys.executable), "developer"),
    )
    captured = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("sim.installation.cli.subprocess.run", fake_run)
    assert _dispatch("study", arguments, paths=paths, workspace_path=None) == 0
    assert captured == [[sys.executable, "-m", module, *arguments]]


def test_wrong_platform_and_source_version_are_rejected(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    wrong_platform = "Windows" if platform.system() != "Windows" else "Linux"
    manifest, _ = _release(tmp_path, "0.25.0", private, artifact_platform=wrong_platform)
    with pytest.raises(ContractError, match="no source artifact"):
        install_release(manifest, paths=_paths(tmp_path), public_keys=keys, create_runtime=False)

    manifest, _ = _release(tmp_path / "mismatch", "0.26.0", private)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["version"] = "0.27.0"
    payload = sign_payload(payload, private)  # type: ignore[arg-type]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="does not match signed manifest"):
        install_release(manifest, paths=_paths(tmp_path / "mismatch"), public_keys=keys, create_runtime=False)


@pytest.mark.parametrize("defect", [None, "wheel", "manifest", "extra"])
def test_online_release_fetches_only_signed_bundle_wheels(
    tmp_path: Path, signing_keys: tuple[object, dict[str, RSAPublicKey]], defect: str | None,
) -> None:
    import hashlib

    private, keys = signing_keys
    manifest_path, _ = _release(tmp_path, "0.32.0", private)
    manifest = json.loads(manifest_path.read_text())
    contents = b"qualified native wheel bytes"
    wheel = {
        "name": "native-0.1-py3-none-any.whl", "kind": "wheel",
        "path": "wheelhouse/native-0.1-py3-none-any.whl", "bytes": len(contents),
        "sha256": hashlib.sha256(contents).hexdigest(), "media_type": "application/zip",
    }
    manifest["artifacts"].append(wheel)
    manifest["architecture"] = platform.machine()
    manifest["supply_chain"] = {"offline_runtime_qualification": {"python": platform.python_version()}}
    manifest = sign_payload(manifest, private)  # type: ignore[arg-type]
    manifest_path.write_text(json.dumps(manifest))
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    bundle = manifest_path.parent / f"oel-public-0.32.0-{platform.machine().lower()}-{tag}.bundle.zip"
    bundled = dict(manifest)
    if defect == "manifest":
        bundled["version"] = "0.31.0"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("release-manifest.json", json.dumps(bundled))
        archive.writestr(wheel["path"], b"tampered" if defect == "wheel" else contents)
        if defect == "extra":
            archive.writestr("wheelhouse/unlisted.whl", b"untrusted")
    if defect:
        with pytest.raises((ContractError, ValueError)):
            download_release(manifest_path.as_uri(), paths=_paths(tmp_path), public_keys=keys, allow_local_file=True)
        assert not (_paths(tmp_path).cache / "public/stable/0.32.0/wheelhouse").exists()
    else:
        receipt = download_release(manifest_path.as_uri(), paths=_paths(tmp_path), public_keys=keys, allow_local_file=True)
        assert (Path(receipt["wheelhouse"]) / wheel["name"]).read_bytes() == contents
        assert verify_payload(json.loads(Path(receipt["manifest"]).read_text()), keys)


def test_bundled_online_download_rejects_wrong_minor_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def network(*args: object, **kwargs: object) -> None:
        pytest.fail("A mismatched runtime must fail before downloading or pip")
    monkeypatch.setattr("sim.installation.manager._download_url", network)
    manifest = {
        "version": "0.32.0", "artifacts": [{"kind": "wheel"}],
        "supply_chain": {"offline_runtime_qualification": {"python": f"3.{sys.version_info.minor + 1}.0"}},
    }
    with pytest.raises(ContractError, match="qualified for CPython"):
        download_release_wheelhouse(manifest, "https://example.test/release-manifest.json", tmp_path)


def test_bootstrap_selects_signed_qualified_python_before_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.installers import bootstrap_install

    monkeypatch.setattr(bootstrap_install.sys, "version_info", (3, 14, 0))
    monkeypatch.setattr(bootstrap_install.shutil, "which", lambda name: "/trusted/python3.11" if name == "python3.11" else None)
    monkeypatch.setattr(bootstrap_install.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    calls = []
    def execute(path: str, arguments: list[str]) -> None:
        calls.append((path, arguments))
        raise RuntimeError("verified interpreter selected")
    monkeypatch.setattr(bootstrap_install.os, "execv", execute)
    manifest = {"artifacts": [{"kind": "wheel"}], "supply_chain": {"offline_runtime_qualification": {"python": "3.11.14"}}}
    with pytest.raises(RuntimeError, match="verified interpreter selected"):
        bootstrap_install._use_qualified_python(manifest)
    assert calls[0][0] == "/trusted/python3.11"
    monkeypatch.setattr(bootstrap_install.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="requires CPython 3.11"):
        bootstrap_install._use_qualified_python(manifest)


def test_download_preserves_signed_manifest_and_rejects_feed_rollback(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    manifest_path, _ = _release(tmp_path, "0.26.0", private)
    paths = _paths(tmp_path)
    receipt = download_release(manifest_path.as_uri(), paths=paths, public_keys=keys, allow_local_file=True)
    downloaded = json.loads(Path(receipt["manifest"]).read_text(encoding="utf-8"))
    assert verify_payload(downloaded, keys)
    install_release(receipt["manifest"], paths=paths, public_keys=keys, create_runtime=False)

    channel = {
        "schema_version": CHANNEL_INDEX_SCHEMA,
        "edition": "public",
        "channel": "stable",
        "latest": "0.26.0",
        "manifest_url": manifest_path.as_uri(),
        "published_at": "2026-08-15T00:00:00Z",
    }
    channel_path = tmp_path / "channel.json"
    channel_path.write_text(json.dumps(sign_payload(channel, private)), encoding="utf-8")  # type: ignore[arg-type]
    configured = configure_channel(
        channel_path.as_uri(),
        paths=paths,
        source="test-bootstrap",
        allow_local_file=True,
    )
    assert Path(configured["path"]).is_file()
    assert configured_channel_url(paths=paths, allow_local_file=True) == channel_path.as_uri()
    check_channel(public_keys=keys, paths=paths, allow_local_file=True)
    channel["latest"] = "0.25.0"
    channel_path.write_text(json.dumps(sign_payload(channel, private)), encoding="utf-8")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="rollback rejected"):
        check_channel(public_keys=keys, paths=paths, allow_local_file=True)


def test_download_cache_isolated_by_edition_and_channel(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    stable_manifest, _ = _release(tmp_path / "stable", "0.26.0", private)
    preview_manifest, _ = _release(tmp_path / "preview", "0.26.0", private)
    preview_payload = json.loads(preview_manifest.read_text(encoding="utf-8"))
    preview_payload.pop("signature")
    preview_payload["channel"] = "preview"
    preview_manifest.write_text(
        json.dumps(sign_payload(preview_payload, private)),  # type: ignore[arg-type]
        encoding="utf-8",
    )
    paths = _paths(tmp_path / "managed")

    stable = download_release(stable_manifest.as_uri(), paths=paths, public_keys=keys, allow_local_file=True)
    preview = download_release(preview_manifest.as_uri(), paths=paths, public_keys=keys, allow_local_file=True)

    assert Path(stable["manifest"]).parent != Path(preview["manifest"]).parent
    assert json.loads(Path(stable["manifest"]).read_text(encoding="utf-8"))["channel"] == "stable"
    assert json.loads(Path(preview["manifest"]).read_text(encoding="utf-8"))["channel"] == "preview"


def test_engine_lease_checks_selected_version_while_holding_transaction_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    version_root = paths.version_root("0.26.0")
    version_root.mkdir(parents=True)

    class RemovingLock:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> None:
            shutil.rmtree(version_root)

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("sim.installation.cli.StateLock", RemovingLock)

    with pytest.raises(RuntimeError, match="disappeared before it could be leased"):
        with _engine_lease(paths, "0.26.0", "official"):
            pytest.fail("missing selected engine must not be dispatched")


def test_install_latest_uses_configured_channel_without_activation(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    manifest_path, _ = _release(tmp_path, "0.26.0", private)
    channel = {
        "schema_version": CHANNEL_INDEX_SCHEMA,
        "edition": "public",
        "channel": "stable",
        "latest": "0.26.0",
        "manifest_url": manifest_path.as_uri(),
        "published_at": "2026-08-15T00:00:00Z",
    }
    channel_path = tmp_path / "channel.json"
    channel_path.write_text(json.dumps(sign_payload(channel, private)), encoding="utf-8")  # type: ignore[arg-type]
    paths = _paths(tmp_path / "managed")
    configure_channel(
        channel_path.as_uri(),
        paths=paths,
        source="test-bootstrap",
        allow_local_file=True,
    )

    receipt = install_latest_release(
        paths=paths,
        public_keys=keys,
        create_runtime=False,
        allow_local_file=True,
    )

    assert receipt["status"] == "ready"
    assert receipt["version"] == "0.26.0"
    assert receipt["activated"] is False
    assert receipt["workspace_modified"] is False
    assert not paths.current_state.exists()
    assert verify_installation("0.26.0", paths=paths, full=True)["status"] == "official"

    channel["latest"] = "0.27.0"
    channel["published_at"] = "2026-08-16T00:00:00Z"
    channel_path.write_text(json.dumps(sign_payload(channel, private)), encoding="utf-8")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="identities do not agree"):
        install_latest_release(
            paths=paths,
            public_keys=keys,
            create_runtime=False,
            allow_local_file=True,
        )


def test_channel_configuration_requires_https(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="must use HTTPS"):
        configure_channel("http://downloads.example.test/stable/channel.json", paths=_paths(tmp_path))


def test_template_sync_classifies_user_and_upstream_changes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    template_source = tmp_path / "quickstart.yaml"
    template_source.write_text("scenario_name: original\n", encoding="utf-8")
    init_workspace(workspace_root, engine_version="0.25.0", quickstart_config=template_source)
    current = workspace_root / "configs" / "quickstart_5min.yaml"
    current.write_text(current.read_text(encoding="utf-8") + "# user edit\n", encoding="utf-8")
    target_root = tmp_path / "new-template"
    target_file = target_root / "configs" / "quickstart_5min.yaml"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("scenario_name: upstream\nschema_version: oel.scenario.v1\n", encoding="utf-8")
    target_manifest = tmp_path / "target-template.json"
    target_manifest.write_text(
        json.dumps(
            {
                "schema_version": "oel.template-manifest.v1",
                "template_id": "oel.workspace.default.v2",
                "files": [
                    {
                        "path": "configs/quickstart_5min.yaml",
                        "sha256": sha256_file(target_file),
                        "user_editable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    plan = plan_template_sync(
        workspace_root,
        target_template_manifest=target_manifest,
        template_root=target_root,
    )

    assert plan["status"] == "manual_review"
    changes = {row["path"]: row for row in plan["changes"]}
    assert changes["configs/quickstart_5min.yaml"]["classification"] == "conflict"
    assert changes["AGENTS.md"]["classification"] == "upstream_removed"
    assert current.read_text(encoding="utf-8").endswith("# user edit\n")


def test_release_build_is_reproducible_signed_and_contains_evidence(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private, keys = signing_keys
    source = tmp_path / "public-source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "orbital-engineering-lab"\nversion = "0.25.0"\n',
        encoding="utf-8",
    )
    (source / "README.md").write_text("public fixture\n", encoding="utf-8")
    (source / "constraints").mkdir()
    (source / "constraints" / "py314.txt").write_text("example==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    private_path = tmp_path / "private-key.json"
    private_path.write_text(json.dumps(private_key_to_json(private)), encoding="utf-8")  # type: ignore[arg-type]
    public_path = tmp_path / "public-keys.json"
    public_path.write_text(
        json.dumps({"keys": [public_key_to_json(next(iter(keys.values())))]}),
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture/__init__.py", "")
        archive.writestr("fixture-1.0.dist-info/METADATA", "Name: fixture\nVersion: 1.0\n")
        archive.writestr("fixture-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr("fixture-1.0.dist-info/RECORD", "")
    from tools.build_installable_release import release_source_identity
    from tools.native_source_identity import native_source_identity

    native_receipts = []
    native_inventory = []
    for name, crate_name in (("oel-rust-orbit", "oel-orbit"), ("oel-rust-game", "oel-game")):
        crate = source / "rust" / crate_name
        (crate / "src").mkdir(parents=True)
        for file in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "src/lib.rs"):
            (crate / file).write_text("# public native fixture\n", encoding="utf-8")
        native_wheel = wheelhouse / f"{name.replace('-', '_')}-1.0-py3-none-any.whl"
        with zipfile.ZipFile(native_wheel, "w") as archive:
            archive.writestr(f"{name.replace('-', '_')}-1.0.dist-info/METADATA", f"Name: {name}\nVersion: 1.0\n")
            archive.writestr(f"{name.replace('-', '_')}-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
            archive.writestr(f"{name.replace('-', '_')}-1.0.dist-info/RECORD", "")
        native_receipts.append({
            "package": name, "wheel": native_wheel.name, "sha256": sha256_file(native_wheel),
            "source_identity_sha256": native_source_identity(crate)["sha256"],
        })
        native_inventory.append({"artifact": native_wheel.name, "artifact_type": "wheel", "sha256": sha256_file(native_wheel)})
    evidence_payloads = {
        "pip-install-report.json": '{"install": []}\n',
        "build-install-report.json": '{"install": []}\n',
        "pip-check.txt": "No broken requirements found.\n",
        "wheel-inventory.json": json.dumps(
            {
                "packages": [*native_inventory,
                    {
                        "artifact": wheel.name,
                        "artifact_type": "wheel",
                        "sha256": sha256_file(wheel),
                    }
                ]
            }
        )
        + "\n",
        "sbom.cdx.json": '{"private_path": "/Users/example/private-oel", "product": "oel-pro"}\n',
        "python-freeze.txt": "fixture==1.0\n",
        "pip-audit.json": '{"dependencies": []}\n',
    }
    evidence_artifacts = []
    for name, content in evidence_payloads.items():
        artifact = evidence / name
        artifact.write_text(content, encoding="utf-8")
        evidence_artifacts.append(
            {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}
        )
    gate = {
        "schema_version": 1,
        "kind": "oel_supply_chain_gate",
        "product": "oel-public",
        "isolated_environment": True,
        "source_identity": release_source_identity(source),
        "dependency_sources": {"qualified_native_wheels": native_receipts},
        "package_version": "0.25.0",
        "passed": True,
        "artifacts": evidence_artifacts,
    }
    (evidence / "supply-chain-gate.json").write_text(json.dumps(gate), encoding="utf-8")
    qualification = {
        "status": "passed",
        "network_used": False,
        "profile": "full",
        "python": "3.14.0",
        "platform": "fixture",
        "architecture": "fixture",
    }
    monkeypatch.setattr(
        "tools.build_installable_release._verify_offline_runtime_install",
        lambda **_kwargs: qualification,
    )
    monkeypatch.setattr("tools.check_public_export.check_public_export", lambda _root: [])

    first_output = tmp_path / "build-one"
    stale_evidence = first_output / "release-evidence" / "supply-chain-gate.json"
    stale_evidence.parent.mkdir(parents=True)
    stale_evidence.write_text('{"private": true}\n', encoding="utf-8")
    first = build_release(
        source_root=source,
        output_dir=first_output,
        edition="public",
        channel="stable",
        version="0.25.0",
        private_key=private_path,
        public_keys=public_path,
        base_url="https://downloads.example.test/oel/0.25.0",
        channel_url="https://downloads.example.test/oel/stable/channel.json",
        developer_unsigned=False,
        epoch=315532800,
        supply_chain_evidence=evidence,
        wheelhouse=wheelhouse,
    )
    second = build_release(
        source_root=source,
        output_dir=tmp_path / "build-two",
        edition="public",
        channel="stable",
        version="0.25.0",
        private_key=private_path,
        public_keys=public_path,
        base_url="https://downloads.example.test/oel/0.25.0",
        channel_url="https://downloads.example.test/oel/stable/channel.json",
        developer_unsigned=False,
        epoch=315532800,
        supply_chain_evidence=evidence,
        wheelhouse=wheelhouse,
    )

    assert sha256_file(first["artifact"]) == sha256_file(second["artifact"])
    manifest = json.loads(Path(first["manifest"]).read_text(encoding="utf-8"))
    assert verify_payload(manifest, keys)
    assert "full" in manifest["profiles"]
    assert manifest["supply_chain"]["status"] == "passed"
    assert manifest["supply_chain"]["offline_runtime_qualification"] == qualification
    assert Path(first["offline_bundle"]).name.endswith("-py314.bundle.zip")
    assert {row["name"] for row in first["release_assets"]} == {
        "SHA256SUMS",
        "bootstrap_install.py",
        "install.ps1",
        "install.sh",
        Path(first["offline_bundle"]).name,
        "orbital-engineering-lab-0.25.0-public.tar.gz",
        "public-stable.json",
        "release-manifest.json",
        "trusted-release-keys.json",
    }
    assert manifest["supply_chain"]["gate"] == "release-evidence/public-supply-chain-attestation.json"
    assert [item["name"] for item in manifest["supply_chain"]["artifacts"]] == [
        "public-supply-chain-attestation.json"
    ]
    assert "__OEL_BOOTSTRAP_SHA256__" not in Path(first["installers"][1]).read_text(encoding="utf-8")
    installer_text = Path(first["installers"][1]).read_text(encoding="utf-8")
    assert "__OEL_DEFAULT_CHANNEL_URL__" not in installer_text
    assert "https://downloads.example.test/oel/stable/channel.json" in installer_text
    assert first["channel_url"] == "https://downloads.example.test/oel/stable/channel.json"
    offline = install_bundle(
        first["offline_bundle"],
        paths=_paths(tmp_path / "offline-install"),
        public_keys=keys,
        create_runtime=False,
    )
    assert offline["installation"]["status"] == "official"
    assert offline["effects"]["network_used"] is False
    with zipfile.ZipFile(first["offline_bundle"]) as archive:
        names = set(archive.namelist())
        assert "release-evidence/public-supply-chain-attestation.json" in names
        assert "release-evidence/supply-chain-gate.json" not in names
        assert "release-evidence/sbom.cdx.json" not in names
        attestation_bytes = archive.read("release-evidence/public-supply-chain-attestation.json")
    attestation_text = attestation_bytes.decode("utf-8")
    assert str(tmp_path) not in attestation_text
    assert "/Users/example/private-oel" not in attestation_text
    assert "oel-pro" not in attestation_text
    attestation = json.loads(attestation_text)
    evidence_paths = [
        evidence / "supply-chain-gate.json",
        *[evidence / name for name in evidence_payloads],
    ]
    expected_source_evidence = [
        {
            "bytes": artifact.stat().st_size,
            "name": artifact.name,
            "sha256": sha256_file(artifact),
        }
        for artifact in evidence_paths
    ]
    assert attestation == {
        "package_version": "0.25.0",
        "schema_version": "oel.public-supply-chain-attestation.v1",
        "source_evidence": expected_source_evidence,
        "status": "passed",
    }
    assert "build-install-report.json" in {row["name"] for row in attestation["source_evidence"]}


def test_developer_unsigned_build_does_not_require_trusted_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "public-source"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "orbital-engineering-lab"\nversion = "0.27.0"\n',
        encoding="utf-8",
    )
    (source / "README.md").write_text("public fixture\n", encoding="utf-8")
    monkeypatch.setattr("tools.check_public_export.check_public_export", lambda _root: [])

    receipt = build_release(
        source_root=source,
        output_dir=tmp_path / "unsigned",
        edition="public",
        channel="preview",
        version="0.27.0",
        private_key=None,
        public_keys=None,
        base_url=None,
        developer_unsigned=True,
        epoch=315532800,
        channel_url=None,
    )

    assert receipt["status"] == "ready"
    assert receipt["developer_unsigned"] is True
    assert "trusted-release-keys.json" not in {row["name"] for row in receipt["release_assets"]}


def test_offline_runtime_qualification_requires_matching_python_minor() -> None:
    matching = {
        "supply_chain": {
            "offline_runtime_qualification": {"python": platform.python_version()},
        }
    }
    _validate_offline_runtime_compatibility(matching)

    other_minor = 10 if sys.version_info.minor != 10 else 11
    mismatched = {
        "supply_chain": {
            "offline_runtime_qualification": {"python": f"3.{other_minor}.0"},
        }
    }
    with pytest.raises(ContractError, match="Offline bundle was qualified for CPython"):
        _validate_offline_runtime_compatibility(mismatched)


def test_official_signing_material_requires_strong_matching_trust_root(tmp_path: Path) -> None:
    weak = generate_rsa_private_key("weak", bits=512)
    weak_private = tmp_path / "weak-private.json"
    weak_public = tmp_path / "weak-public.json"
    weak_private.write_text(json.dumps(private_key_to_json(weak)), encoding="utf-8")
    weak_public.write_text(
        json.dumps({"keys": [public_key_to_json(RSAPublicKey(key_id=weak.key_id, n=weak.n))]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 2048"):
        _official_signing_material(private_key=weak_private, public_keys=weak_public)

    strong = generate_rsa_private_key("strong", bits=2048)
    strong_private = tmp_path / "strong-private.json"
    mismatched_public = tmp_path / "mismatched-public.json"
    strong_private.write_text(json.dumps(private_key_to_json(strong)), encoding="utf-8")
    mismatched_public.write_text(
        json.dumps({"keys": [public_key_to_json(RSAPublicKey(key_id=strong.key_id, n=weak.n))]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        _official_signing_material(private_key=strong_private, public_keys=mismatched_public)


def test_release_source_inventory_excludes_local_drag_racing_prototype(tmp_path: Path) -> None:
    source = tmp_path / "source"
    tracked = source / "sim" / "game" / "runner.py"
    local_config = source / "sim" / "game" / "configs" / "game_training_rpo_bonus_drag_racing.yaml"
    local_art = source / "sim" / "game" / "assets" / "drag_racing_chaser_side.png"
    local_test = source / "sim" / "tests" / "test_game_drag_racing_local.py"
    for path in (tracked, local_config, local_art, local_test):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    relative_files = {path.relative_to(source).as_posix() for path in _source_files(source)}

    assert relative_files == {"sim/game/runner.py"}


def test_release_source_inventory_uses_only_tracked_files_in_git_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    tracked = source / "tracked.py"
    ignored = source / "local-only.txt"
    ignore_file = source / ".gitignore"
    tracked.write_text("tracked = True\n", encoding="utf-8")
    ignored.write_text("not release material\n", encoding="utf-8")
    ignore_file.write_text("local-only.txt\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", ".gitignore", "tracked.py"], check=True)

    relative_files = {path.relative_to(source).as_posix() for path in _source_files(source)}

    assert relative_files == {".gitignore", "tracked.py"}


def test_wheelhouse_rejects_non_wheel_bytes(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "fixture-1.0-py3-none-any.whl").write_bytes(b"not a wheel")
    with pytest.raises(ValueError, match="not a valid wheel archive"):
        _collect_wheelhouse(wheelhouse, tmp_path / "output", required=True)


def test_offline_runtime_qualification_uses_no_index_and_import_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.build_installable_release.subprocess.run", fake_run)
    result = _verify_offline_runtime_install(
        archive=tmp_path / "release.tar.gz",
        wheelhouse=tmp_path / "wheelhouse",
        version="0.26.0",
    )

    assert result["status"] == "passed"
    assert result["network_used"] is False
    assert "--no-index" in commands[1]
    assert "--only-binary=:all:" in commands[1]
    assert commands[1][-1].endswith("release.tar.gz[full]")
    assert "import sim.installation.cli" in commands[2][-1]


def test_rendered_posix_installer_routes_verified_bootstrap_arguments(tmp_path: Path) -> None:
    keys = tmp_path / "keys.json"
    keys.write_text('{"keys": []}\n', encoding="utf-8")
    installers = _render_installers(
        tmp_path / "rendered",
        public_keys=keys,
        base_url="https://downloads.example.test/oel/0.26.0",
        channel_url="https://downloads.example.test/oel/stable/channel.json",
    )
    bootstrap, install_sh, _install_ps1 = installers
    syntax = subprocess.run(["sh", "-n", str(install_sh)], capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3.14"
    fake_python.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OEL_TEST_RECORD\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/bin/sh\n"
        "out=''\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then out=$2; shift 2; else shift; fi\n"
        "done\n"
        "cp \"$OEL_TEST_BOOTSTRAP\" \"$out\"\n",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    record = tmp_path / "posix-arguments.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "OEL_TEST_BOOTSTRAP": str(bootstrap),
        "OEL_TEST_RECORD": str(record),
        "TMPDIR": str(tmp_path),
    }
    completed = subprocess.run(
        ["sh", str(install_sh), "--data-root", str(tmp_path / "data")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = record.read_text(encoding="utf-8")
    assert "--manifest-url\nhttps://downloads.example.test/oel/0.26.0/release-manifest.json" in arguments
    assert "--channel-url\nhttps://downloads.example.test/oel/stable/channel.json" in arguments
    assert f"--data-root\n{tmp_path / 'data'}" in arguments


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell is not installed on this host")
def test_rendered_powershell_installer_routes_verified_bootstrap_arguments(tmp_path: Path) -> None:
    keys = tmp_path / "keys.json"
    keys.write_text('{"keys": []}\n', encoding="utf-8")
    bootstrap, _install_sh, install_ps1 = _render_installers(
        tmp_path / "rendered",
        public_keys=keys,
        base_url="https://downloads.example.test/oel/0.26.0",
        channel_url="https://downloads.example.test/oel/stable/channel.json",
    )
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_py = fake_bin / "py"
    fake_py.write_text(
        "#!/bin/sh\n"
        "if [ \"${2:-}\" = \"-c\" ]; then exit 0; fi\n"
        "printf '%s\\n' \"$@\" > \"$OEL_TEST_RECORD\"\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    record = tmp_path / "powershell-arguments.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
        "OEL_TEST_BOOTSTRAP": str(bootstrap),
        "OEL_TEST_INSTALLER": str(install_ps1),
        "OEL_TEST_RECORD": str(record),
    }
    command = (
        "function global:Invoke-WebRequest { param($UseBasicParsing, $Uri, $OutFile) "
        "Copy-Item -LiteralPath $env:OEL_TEST_BOOTSTRAP -Destination $OutFile }; "
        "& $env:OEL_TEST_INSTALLER --data-root test-data"
    )
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", command],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    arguments = record.read_text(encoding="utf-8")
    assert "--manifest-url\nhttps://downloads.example.test/oel/0.26.0/release-manifest.json" in arguments
    assert "--channel-url\nhttps://downloads.example.test/oel/stable/channel.json" in arguments


def test_trusted_key_rotation_and_sanitized_support_receipt(
    tmp_path: Path,
    signing_keys: tuple[object, dict[str, RSAPublicKey]],
) -> None:
    private, keys = signing_keys
    paths = _paths(tmp_path)
    paths.ensure()
    paths.trusted_release_keys.write_text(
        json.dumps({"keys": [public_key_to_json(next(iter(keys.values())))]}),
        encoding="utf-8",
    )
    successor = generate_rsa_private_key("successor", bits=2048)
    registry = sign_payload(
        {
            "schema_version": "oel.trusted-key-registry.v1",
            "published_at": "2026-08-19T00:00:00Z",
            "keys": [public_key_to_json(RSAPublicKey(key_id=successor.key_id, n=successor.n))],
        },
        private,  # type: ignore[arg-type]
    )
    registry_path = tmp_path / "new-keys.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    rotation = rotate_trusted_release_keys(registry_path, paths=paths, current_keys=keys)
    assert rotation["key_ids"] == ["successor"]
    receipt_path = tmp_path / "support.json"
    support = write_support_receipt(receipt_path, paths=paths)
    serialized = receipt_path.read_text(encoding="utf-8")
    assert support["receipt"]["privacy"]["user_source_included"] is False
    assert str(tmp_path) not in serialized


def test_workspace_includes_hash_bound_relative_study_example(tmp_path: Path) -> None:
    from sim.installation.resources import resource_path

    source = resource_path("configs", "acceptance_relative_coast.yaml")
    root = tmp_path / "study"
    init_workspace(root, engine_version="0.28.0", example_configs=(source,))
    target = root / "configs" / source.name
    raw = yaml.safe_load(target.read_text())
    assert raw["scenario_name"] == "acceptance_relative_coast"
    assert raw["outputs"]["output_dir"] == "outputs/acceptance_relative_coast"
    receipt = json.loads((root / ".oel/template-manifest.json").read_text())
    assert receipt["files"] == [
        {"path": "AGENTS.md", "sha256": sha256_file(root / "AGENTS.md"), "user_editable": True},
        {"path": "configs/acceptance_relative_coast.yaml", "sha256": sha256_file(target), "user_editable": True},
    ]
