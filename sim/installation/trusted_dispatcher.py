"""External preflight dispatcher used by generated OEL launchers.

The installer copies this file and the exact signed-contract helpers into an
operator-owned directory outside managed version roots.  It runs with the
operator's Python in isolated mode, authenticates the selected version, and
only then starts that version's managed interpreter.  Keep imports here
stdlib-only until the trusted bundle has been placed on ``sys.path``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    # Forwarded OEL CLI flags may intentionally resemble the launcher's own
    # options (for example ``--config`` versus ``--config-root``). Never let
    # argparse abbreviation consume a user argument before the ``--`` seam.
    parser = argparse.ArgumentParser(prog="oel-trusted-dispatcher", allow_abbrev=False)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle_root = Path(__file__).resolve().parent / "trusted-verifier"
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        print("OEL launcher trusted verifier bundle is missing.", file=sys.stderr)
        return 78
    required_bundle_paths = (
        bundle_root / "sim",
        bundle_root / "sim" / "__init__.py",
        bundle_root / "sim" / "schema_versions.py",
        bundle_root / "sim" / "installation",
        bundle_root / "sim" / "installation" / "__init__.py",
        bundle_root / "sim" / "installation" / "contracts.py",
        bundle_root / "sim" / "installation" / "signing.py",
        bundle_root / "sim" / "installation" / "integrity.py",
        bundle_root / "sim" / "installation" / "integrity_transport.py",
        bundle_root / "sim" / "installation" / "trusted_verifier.py",
    )
    if any(path.is_symlink() or not path.exists() for path in required_bundle_paths):
        print("OEL launcher trusted verifier bundle is incomplete or linked.", file=sys.stderr)
        return 78
    # The bundle is copied outside managed versions by the operator-controlled
    # installer.  Inserting only this directory after isolated startup avoids
    # importing a mutable managed ``sim`` package during preflight.
    sys.path.insert(0, str(bundle_root))
    try:
        from sim.installation.trusted_verifier import verify_official_installation

        verified = verify_official_installation(
            args.version,
            data_root=args.data_root,
            config_root=args.config_root,
        )
    except Exception as exc:  # fail closed before any managed import/exec
        print(f"OEL launcher preflight refused: {exc}", file=sys.stderr)
        return 78

    source = Path(str(verified["source"]))
    python = Path(str(verified["python"]))
    if not python.is_file() and not python.is_symlink():
        print("OEL launcher managed interpreter disappeared after preflight.", file=sys.stderr)
        return 78
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    environment = dict(os.environ)
    environment["OEL_ENGINE_VERSION"] = str(verified["version"])
    environment["OEL_INSTALLATION_DISPOSITION"] = "official"
    environment["OEL_MANAGED_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve(strict=False))
    environment["OEL_MANAGED_CONFIG_ROOT"] = str(Path(args.config_root).expanduser().resolve(strict=False))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    # Do not let caller-controlled Python environment settings redirect the
    # first managed import or a bytecode-cache lookup outside the verified
    # source/runtime roots.
    for name in (
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONUSERBASE",
        "PYTHONEXECUTABLE",
        "PYTHONPYCACHEPREFIX",
    ):
        environment.pop(name, None)
    # Ignore an incidental caller PYTHONPATH.  The source path was just
    # authenticated and is inserted by the isolated stdlib bootstrap below.
    # Keep the caller's working directory intact so relative config and
    # workspace paths retain the CLI's documented meaning.
    environment.pop("PYTHONPATH", None)
    bootstrap = (
        "import runpy,sys;"
        "source=sys.argv.pop(1);"
        "sys.path.insert(0,source);"
        "runpy.run_module('sim.installation.cli',run_name='__main__')"
    )
    command = [
        str(python),
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(source),
        "--data-root",
        str(args.data_root),
        "--config-root",
        str(args.config_root),
        *arguments,
    ]
    # This is the first managed-runtime execution in the launcher path.  The
    # signed source/runtime checks above happen in the external bundle first.
    os.execve(str(python), command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
