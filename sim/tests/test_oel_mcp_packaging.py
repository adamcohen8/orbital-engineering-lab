from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _effective_dependency_site_packages() -> tuple[Path, ...]:
    """Return dependency roots actually visible to this test interpreter.

    ``site.getsitepackages()`` omits site-package directories injected by a
    processed ``.pth`` file.  The child venv processes the generated fixture
    ``.pth`` only once, so copy the already-effective path entries instead of
    expecting it to recursively process the parent venv's ``.pth`` files.
    """

    roots: list[Path] = []
    for raw in [*site.getsitepackages(), *sys.path]:
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if not candidate.is_dir() or not any(
            part in {"site-packages", "dist-packages"} for part in candidate.parts
        ):
            continue
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def test_noneditable_wheel_launches_trainer_help_outside_checkout(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    subprocess.run(
        [sys.executable, "-c",
         "from setuptools.build_meta import build_wheel; import sys; "
         "build_wheel(sys.argv[1], config_settings={"
         "'--global-option': ['egg_info', '--egg-base', sys.argv[2], 'build', '--build-base', sys.argv[3]], "
         "'--build-option': ['--bdist-dir', sys.argv[4]]})",
         str(wheel_dir), str(metadata_dir), str(tmp_path / "build"), str(tmp_path / "bdist")],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "run_game.py" in archive.namelist()
    environment = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    installed_site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import site; print(site.getsitepackages()[0])"], text=True,
    ).strip())
    # Reuse the test environment's dependencies, while OEL itself must come
    # from the newly installed wheel at the front of site-packages.
    (installed_site / "test-dependencies.pth").write_text(
        "".join(f"{path}\n" for path in _effective_dependency_site_packages()),
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH" and not key.startswith("OEL_")}
    env.update(OEL_DATA_HOME=str(tmp_path / "data"), OEL_CONFIG_HOME=str(tmp_path / "config"))
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(wheel)],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    identity = json.loads(subprocess.check_output(
        [str(python), "-I", "-c", "import json,sim,run_game; print(json.dumps([sim.__file__,run_game.__file__]))"],
        cwd=tmp_path, env=env, text=True,
    ))
    assert all(Path(path).is_relative_to(installed_site) for path in identity)
    result = subprocess.run(
        [str(python), "-I", "-m", "sim.installation.cli", "trainer", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--controlled-object" in result.stdout


@pytest.mark.external
def test_m5_wheel_packages_supported_mcp_profiles_and_keeps_dependency_optional(
    tmp_path: Path, monkeypatch
) -> None:
    import anyio
    from mcp import Client, StdioServerParameters, stdio_client

    monkeypatch.chdir(ROOT)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [sys.executable, "-c",
         "from setuptools.build_meta import build_wheel; import sys; build_wheel(sys.argv[1])",
         str(wheel_dir)],
        check=True,
    )
    wheel_name = next(wheel_dir.glob("*.whl")).name

    with zipfile.ZipFile(wheel_dir / wheel_name) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
        wheel_metadata_name = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        metadata = archive.read(metadata_name).decode("utf-8")
        entry_points = archive.read(entry_points_name).decode("utf-8")
        wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")

    assert wheel_name.endswith("-py3-none-any.whl")
    assert "Root-Is-Purelib: true" in wheel_metadata
    assert "Tag: py3-none-any" in wheel_metadata
    assert "License-Expression: Apache-2.0" in metadata
    assert "License-File: LICENSE.txt" in metadata
    assert any(name.startswith("sim/") for name in names)
    assert "integrations/oel_mcp/acceptance.py" in names
    assert "integrations/oel_mcp/diagnostics.py" in names
    assert "integrations/oel_mcp/public_server.py" in names
    assert "integrations/oel_mcp/reporting.py" in names
    assert "integrations/oel_mcp/sdk_protocol.py" in names
    assert "integrations/oel_mcp/execution.py" in names
    assert "integrations/oel_mcp/resources.py" in names
    assert "integrations/oel_mcp/resource_data/operator-guide.md" in names
    assert "sim/installation/cli.py" in names
    assert "sim/installation/manager.py" in names
    assert "sim/installation/schemas/channel-config.schema.json" in names
    assert "sim/installation/data/agent-bootstrap.md" in names
    assert "sim/installation/schemas/release-manifest.schema.json" in names
    assert "sim/schema_versions.py" in names
    assert any(name.endswith(".data/data/share/oel/configs/quickstart_5min.yaml") for name in names)
    pro_modules = {
        "integrations/oel_mcp/pro_handlers.py",
        "integrations/oel_mcp/pro_acceptance.py",
        "integrations/oel_mcp/pro_diagnostics.py",
        "integrations/oel_mcp/pro_registry.py",
        "integrations/oel_mcp/server.py",
    }
    if (ROOT / "integrations" / "oel_mcp" / "pro_handlers.py").is_file():
        assert pro_modules <= names
    else:
        assert not pro_modules & names
    mcp_requirements = [line for line in metadata.splitlines() if line.startswith("Requires-Dist: mcp")]
    assert mcp_requirements == [
        'Requires-Dist: mcp<3,>=2.0.0; extra == "mcp"',
        'Requires-Dist: mcp<3,>=2.0.0; extra == "full"',
    ]
    assert "oel-mcp = integrations.oel_mcp.public_server:main" in entry_points
    assert "oel = sim.installation.cli:main" in entry_points
    environment = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    entrypoint = scripts / ("oel-mcp.exe" if os.name == "nt" else "oel-mcp")
    installed_site = Path(
        subprocess.run(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    (installed_site / "oel-mcp-test-parent-venv.pth").write_text(
        "".join(
            str(Path(path).resolve()) + "\n"
            for path in dict.fromkeys([*site.getsitepackages(), *sys.path])
            if Path(path).name == "site-packages" and Path(path).is_dir()
        ),
        encoding="utf-8",
    )
    installed_env = dict(os.environ)
    installed_env.pop("PYTHONPATH", None)
    installed_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--no-index",
            str(wheel_dir / wheel_name),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=installed_env,
    )
    # Prove the wheel owns imports and seeds its packaged guide outside the checkout.
    identity = subprocess.run(
        [str(python), "-c",
         "import json,sys,sim; from pathlib import Path; "
         "from sim.installation.workspace import init_workspace; "
         "from sim.installation.resources import agent_bootstrap_text; "
         "p=Path(sys.argv[1]); init_workspace(p,engine_version='0.28.0'); "
         "assert (p/'AGENTS.md').read_text()==agent_bootstrap_text(); "
         "print(json.dumps({'sim':sim.__file__,'bootstrap':agent_bootstrap_text()}))",
         str(tmp_path / "installed-workspace")],
        cwd=tmp_path, env=installed_env, check=True, capture_output=True, text=True,
    )
    installed_identity = json.loads(identity.stdout)
    assert Path(installed_identity["sim"]).is_relative_to(installed_site)

    async def exercise() -> tuple[str, tuple[str, ...], str]:
        parameters = StdioServerParameters(
            command=str(entrypoint),
            cwd=tmp_path,
            env={**installed_env, "OEL_MCP_READ_ROOTS": str(tmp_path)},
        )
        async with Client(stdio_client(parameters), mode="auto", cache=None) as client:
            resources = await client.list_resources(cache_mode="reload")
            guide = await client.read_resource("oel://docs/operator-guide/v1", cache_mode="reload")
            bootstrap = await client.read_resource("oel://agent/bootstrap/v1", cache_mode="reload")
            assert bootstrap.contents[0].text == installed_identity["bootstrap"]
            return (
                client.protocol_version,
                tuple(resource.uri for resource in resources.resources),
                str(guide.contents[0].text),
            )

    protocol, resource_uris, guide = anyio.run(exercise)

    assert protocol == "2026-07-28"
    assert resource_uris == (
        "oel://capabilities/tools/v1",
        "oel://review/saved-queries/v1",
        "oel://agent/tasks/v1",
        "oel://docs/operator-guide/v1",
        "oel://handoff/product-kinds/v1",
        "oel://review/plot-recipes/v1",
        "oel://review/animation-recipes/v1",
        "oel://analysis/workflows/v1",

        "oel://agent/bootstrap/v1",
        "oel://agent/workflows/v1",
    )
    assert "supported local stdio OEL MCP surface" in guide
