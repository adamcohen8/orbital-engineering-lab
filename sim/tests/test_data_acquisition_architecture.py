import ast
import json
from pathlib import Path

from sim.installation.cli import _build_parser, _dispatch_commands

ROOT = Path(__file__).resolve().parents[2]


def test_acquisition_owners_do_not_execute_simulations_or_import_mcp():
    forbidden = ("sim.execution", "sim.runtime", "sim.single_run", "integrations.oel_mcp")
    for path in (ROOT / "sim/data_acquisition").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not str(node.module).startswith(forbidden), path
            if isinstance(node, ast.Import):
                assert not any(name.name.startswith(forbidden) for name in node.names), path


def test_launcher_exposes_the_new_command_without_replacing_io():
    assert {"io", "data"} <= set(_dispatch_commands())
    result = _build_parser().parse_args(["data", "providers"])
    assert result.command == "data" and result.arguments == ["providers"]


def test_public_manifest_owns_acquisition_but_never_downloads():
    path = ROOT / "docs/operations/public_surface_manifest.yaml"
    if not path.exists():
        return
    surfaces = json.loads(path.read_text())["public_surfaces"]
    assert "sim/data_acquisition/" in surfaces["public data acquisition"]["owned"]
    assert "acquired-data/" not in str(surfaces["public data acquisition"])


def test_documented_request_examples_satisfy_the_runtime_contract():
    from sim.data_acquisition import Request

    for path in (ROOT / "examples/data_acquisition").glob("*request.json"):
        assert Request.from_dict(json.loads(path.read_text())).provider in {"gravity", "celestrak", "spacetrack"}
