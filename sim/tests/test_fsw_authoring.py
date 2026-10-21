from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sim.fsw_authoring import inspect_candidate
from sim.fsw_authoring.services import (
    describe_capabilities,
    init_candidate,
    plan_workflow,
    run_contract_tests,
    run_smoke,
    validate_candidate_service,
    verify_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _scaffold(root: Path, *, template: str = "adcs", name: str = "public_candidate") -> Path:
    receipt = init_candidate(name, template=template, workspace_root=root)
    assert receipt["status"] == "ready"
    return Path(receipt["result"]["manifest_path"])


def test_public_authoring_capabilities_stop_before_private_verification() -> None:
    capabilities = describe_capabilities()

    assert capabilities["product"] == "OEL Public FSW Authoring Kit"
    assert {item["id"] for item in capabilities["templates"]} == {"adcs", "rpo"}
    assert capabilities["candidate_kinds"] == ["python_stack"]
    assert "controller_bench" in capabilities["private_operations"]
    assert "qualification" in capabilities["private_operations"]
    assert "cfs" + "_sil" in capabilities["private_operations"]
    assert "qualify" not in capabilities["operations"]


def test_public_scaffold_is_safely_inspectable_and_content_bound(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path)

    inspected = inspect_candidate(manifest, workspace_root=tmp_path)
    assert inspected["status"] == "ready"
    assert inspected["safe_inspection"] is True
    assert inspected["candidate_code_imported"] is False
    assert inspected["candidate_code_executed"] is False
    assert inspected["private_operations_available"] is False
    scaffold_receipt = manifest.parent / ".oel/fsw_authoring_scaffold_receipt.json"
    assert verify_receipt(scaffold_receipt, workspace_root=tmp_path)["status"] == "passed"

    plan = plan_workflow(manifest, "smoke", workspace_root=tmp_path)
    assert plan["effects"]["executes"] is True
    assert plan["source_trust_required"] is True
    assert plan["work_order"]["constraints"]["workers"] == 1
    assert plan["work_order"]["constraints"]["network"] is False

    original_hash = inspected["candidate"]["candidate_sha256"]
    stack_path = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    stack_path.write_text(stack_path.read_text(encoding="utf-8") + "\n# revision\n", encoding="utf-8")
    changed = inspect_candidate(manifest, workspace_root=tmp_path)
    assert changed["candidate"]["candidate_sha256"] != original_hash


def test_public_authoring_rejects_simulator_truth_imports_without_importing_candidate(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path)
    stack_path = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    stack_path.write_text(
        stack_path.read_text(encoding="utf-8") + "\nfrom sim.runtime import models as forbidden_truth\n",
        encoding="utf-8",
    )

    receipt = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=False,
        write_receipt=False,
    )

    assert receipt["status"] == "invalid"
    assert receipt["result"]["checks"]["candidate_import"] == "not_run"
    assert receipt["result"]["checks"]["truth_firewall"] == "failed"
    assert any(issue["code"] == "truth_boundary_import" for issue in receipt["issues"])


def test_public_authoring_adcs_lifecycle_tests_smoke_and_receipt_currentness(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path, template="adcs", name="verified_adcs")
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    validated = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=True,
        receipt_dir=validation_dir,
    )
    assert validated["status"] == "ready"
    assert all(value == "passed" for value in validated["result"]["checks"].values())
    validation_id = validated["result"]["validation_id"]

    precreated_test_output = tmp_path / "test-output"
    precreated_test_output.mkdir()
    tested = run_contract_tests(
        manifest,
        workspace_root=tmp_path,
        output_dir=precreated_test_output,
        validation_id=validation_id,
    )
    assert tested["status"] == "passed"
    assert tested["execution"]["returncode"] == 0

    smoked = run_smoke(
        manifest,
        workspace_root=tmp_path,
        output_dir=tmp_path / "smoke-output",
        validation_id=validation_id,
    )
    assert smoked["status"] == "passed"
    assert smoked["summary"]["samples"] == 21
    assert smoked["summary"]["runtime_profile"]["executor"]["object_step_backend"] == "serial"
    assert Path(smoked["summary"]["review_sqlite_path"]).is_file()

    receipt_path = validation_dir / "fsw_validation_receipt.json"
    assert verify_receipt(receipt_path, workspace_root=tmp_path)["status"] == "passed"
    source = next(manifest.parent.joinpath("stacks").glob("*_stack.py"))
    source.write_text(source.read_text(encoding="utf-8") + "\n# stale receipt\n", encoding="utf-8")
    stale = verify_receipt(receipt_path, workspace_root=tmp_path)
    assert stale["status"] == "failed"
    assert stale["candidate_current"] is False


def test_public_rpo_scaffold_passes_trusted_contract_validation(tmp_path: Path) -> None:
    manifest = _scaffold(tmp_path, template="rpo", name="public_rpo")
    receipt = validate_candidate_service(
        manifest,
        workspace_root=tmp_path,
        trusted_import=True,
        write_receipt=False,
    )

    assert receipt["status"] == "ready"
    assert receipt["candidate"]["onboard_contract"] == "oel.fsw.boundary.v1"
    smoke = yaml.safe_load(Path(manifest.parent / "configs/public_rpo_smoke.yaml").read_text(encoding="utf-8"))
    assert smoke["outputs"]["plots"]["enabled"] is False
    assert smoke["outputs"]["animations"]["enabled"] is False
    assert smoke["simulator"]["duration_s"] == 20.0


def test_public_authoring_package_has_no_private_product_dependencies() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "sim/fsw_authoring").rglob("*.py"))
    )

    forbidden = (
        "sim.fswdk",
        "sim.controller_lab",
        "sim.gnc_workbench",
        "sim.licensing",
        "integrations." + "cfs_" + "sil",
        "integrations.oel_mcp",
    )
    assert not any(name in source for name in forbidden)

    schema = json.loads((ROOT / "sim/fsw_authoring/schemas/candidate.schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["kind"]["const"] == "python_stack"
    assert "qualification" not in json.dumps(schema).lower()


@pytest.mark.parametrize("link_kind", ["directory", "file"])
def test_public_scaffold_rejects_symlink_escape_before_writing(tmp_path, link_kind):
    from sim.fsw_authoring.scaffold import scaffold_candidate

    root = tmp_path / "project"
    destination = root / "fsw_candidates" / "safe_candidate"
    destination.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    if link_kind == "directory":
        (destination / "stacks").symlink_to(outside, target_is_directory=True)
    else:
        (destination / "candidate.yaml").symlink_to(outside / "candidate.yaml")
    before = set(destination.rglob("*"))
    with pytest.raises(ValueError, match="inside the authorized workspace"):
        scaffold_candidate("safe_candidate", workspace_root=root)
    assert set(destination.rglob("*")) == before
    assert not list(outside.iterdir())
    if link_kind == "file":
        (destination / "candidate.yaml").unlink()


def test_init_rejects_receipt_directory_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    destination = root / "fsw_candidates/safe_candidate"
    destination.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / ".oel").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="inside the authorized workspace"):
        init_candidate("safe_candidate", workspace_root=root)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("broad_root", [False, True])
def test_candidate_reload_reads_same_size_source_and_helper_edits(tmp_path, monkeypatch, broad_root):
    import importlib
    import os
    import sys

    from sim.fsw_authoring.candidate import clear_candidate_imports

    root = tmp_path / "oel_candidate_reload_probe"
    root.mkdir()
    (root / "__init__.py").write_text("")
    entrypoint = root / "stack.py"
    helper = root / "helper.py"
    module_name = "oel_candidate_reload_probe.stack"
    source_root = tmp_path if broad_root else root
    unrelated = tmp_path / "unrelated_cache.py"
    unrelated.write_text("value = 17\n")
    # This probe deliberately exercises an existing unrelated bytecode cache.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.syspath_prepend(str(tmp_path))
    unrelated_module = importlib.import_module("unrelated_cache")
    unrelated_cache = Path(unrelated_module.__cached__)
    cache_bytes = unrelated_cache.read_bytes()
    try:
        for value in (1, 2):
            entrypoint.write_text("from .helper import helper_value\nvalue = " + str(value) + "\n")
            helper.write_text("helper_value = " + str(value) + "\n")
            for path in (entrypoint, helper):
                os.utime(path, (1700000000, 1700000000))
            clear_candidate_imports(module_name, source_root=source_root)
            module = importlib.import_module(module_name)
            assert (module.value, module.helper_value) == (value, value)
            assert importlib.import_module("unrelated_cache") is unrelated_module
            assert unrelated_cache.read_bytes() == cache_bytes
    finally:
        clear_candidate_imports(module_name, source_root=source_root)
        sys.modules.pop("unrelated_cache", None)


def test_candidate_import_graph_excludes_runtime_packages_inside_broad_root(tmp_path, monkeypatch):
    from sim.fsw_authoring import candidate

    runtime = tmp_path / "embedded_runtime"
    runtime.mkdir()
    (runtime / "installed_helper.py").write_text("value = 1\n")
    (tmp_path / "candidate_probe.py").write_text("import installed_helper\n")
    monkeypatch.syspath_prepend(str(runtime))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(candidate.sysconfig, "get_paths", lambda: {"purelib": str(runtime)})
    sources = candidate._candidate_source_modules("candidate_probe", tmp_path)
    assert sources == {"candidate_probe": tmp_path / "candidate_probe.py"}


@pytest.mark.parametrize("suffixes", [None, frozenset({".py"})])
def test_candidate_tree_hash_preserves_rows_and_rejects_visible_symlinks(tmp_path, suffixes):
    from sim.fsw_authoring.contracts import sha256_file, sha256_tree, sha256_value

    root = tmp_path / "candidate"
    contents = {"a/module.py": "value = 1", "a.py": "value = 2", "notes.txt": "note"}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in (".oel", "__pycache__", ".pytest_cache", ".ruff_cache"):
        ignored = root / "a" / name / "nested.py"
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("ignored source")
    expected = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(root / name for name in contents)
        if suffixes is None or path.suffix.lower() in suffixes
    ]
    digest = sha256_tree(root, suffixes=suffixes)
    assert digest == sha256_value(expected)
    (root / "a" / "module.py").write_text("value = 3")
    assert sha256_tree(root, suffixes=suffixes) != digest
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked_directory").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        sha256_tree(root, suffixes=suffixes)


@pytest.mark.parametrize("permission_denied", [False, True])
def test_candidate_tree_hash_preserves_directory_read_errors(tmp_path, monkeypatch, permission_denied):
    import errno
    import os

    from sim.fsw_authoring.contracts import sha256_tree, sha256_value

    real_scandir = os.scandir

    def unreadable(path):
        if Path(path) == tmp_path:
            if permission_denied:
                raise PermissionError(errno.EACCES, "unreadable", str(path))
            raise OSError(errno.EIO, "device failure", str(path))
        return real_scandir(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "scandir", unreadable)
        if permission_denied:
            assert sha256_tree(tmp_path) == sha256_value([])
        else:
            with pytest.raises(OSError, match="device failure"):
                sha256_tree(tmp_path)


def test_reload_prefers_package_over_same_named_module(tmp_path, monkeypatch):
    import importlib
    import os
    import sys

    from sim.fsw_authoring.candidate import clear_candidate_imports

    name = "candidate_package_precedence_probe"
    package = tmp_path / name
    package.mkdir()
    source = package / "__init__.py"
    (tmp_path / (name + ".py")).write_text("value = 9\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        for value in (1, 2):
            source.write_text("value = " + str(value) + "\n")
            os.utime(source, (1700000000, 1700000000))
            clear_candidate_imports(name, source_root=tmp_path)
            loaded = importlib.import_module(name)
            assert Path(loaded.__file__) == source
            assert loaded.value == value
    finally:
        sys.modules.pop(name, None)
