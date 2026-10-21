from __future__ import annotations

import ast
import json
from pathlib import Path

from sim.data_io.registry import capability_catalog
from sim.installation.cli import _build_parser, _dispatch_commands

ROOT = Path(__file__).resolve().parents[2]


def test_io_owners_do_not_import_simulation_execution_or_runtime_internals():
    forbidden = ("sim.execution", "sim.runtime", "sim.master_simulator", "sim.single_run")
    for path in (ROOT / "sim/data_io").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module).startswith(forbidden), (path, node.module)
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(forbidden) for alias in node.names), path


def test_only_the_separate_fit_owner_imports_the_private_ogp_estimator():
    for path in (ROOT / "sim/data_io").rglob("*.py"):
        if path.name == "fit.py":
            continue
        assert "from sim.estimation.ogp_od" not in path.read_text()
    source = ROOT / "sim/data_io/fit.py"
    if source.is_file():
        code = source.read_text()
        assert "from sim.estimation.ogp_od import fit_ogp_mean_elements_to_observations" in code
        assert "read_review(" in code
        assert "require_pro_feature(FEATURE_TRACKING_OD)" in code
        assert "scipy.optimize" not in code


def test_managed_launcher_discovers_io_and_preserves_the_other_commands():
    assert "io" in _dispatch_commands()
    parsed = _build_parser().parse_args(["io", "formats"])
    assert parsed.command == "io" and parsed.arguments == ["formats"]


def test_public_manifest_owns_modular_io_and_excludes_only_private_fit():
    manifest = ROOT / "docs/operations/public_surface_manifest.yaml"
    if not manifest.is_file():
        # Generated public exports intentionally omit the private export policy.
        return
    value = json.loads(manifest.read_text())
    assert "sim/data_io/" in value["public_surfaces"]["public modular data IO"]["owned"]
    exclusions = (ROOT / "docs/operations/public_export_exclude.txt").read_text().splitlines()
    assert "sim/data_io/fit.py" in exclusions
    assert "sim/tests/test_data_io_fit.py" in exclusions
    assert "sim/data_io/" not in exclusions


def test_capabilities_separate_file_directions_and_optional_backends():
    catalog = capability_catalog()
    rows = {row["format"]: row for row in catalog["formats"]}
    assert rows["sentinel-eof"]["read"] and not rows["sentinel-eof"]["write"]
    assert rows["czml"]["write"] and not rows["czml"]["read"]
    assert rows["spk"]["optional_dependency"] == "spiceypy"
    assert catalog["operations"]["fit-review"]["entitlement"] == "tracking_od"
