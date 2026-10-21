"""Unified end-user launcher and managed-update CLI for OEL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from sim.project_version import installed_project_version, source_project_version

from .contracts import load_json_object
from .manager import (
    activate,
    check_channel,
    cleanup,
    configure_channel,
    download_release,
    install_bundle,
    install_latest_release,
    install_release,
    installation_status,
    keys_from_path,
    pro_installation_available,
    recover_lock,
    rollback,
    rotate_trusted_release_keys,
    uninstall,
    verify_installation,
    write_support_receipt,
)
from .paths import InstallationPaths
from .resources import quickstart_config_path, resource_path
from .state import StateLock, atomic_write_json, atomic_write_text, read_state
from .workspace import (
    WORKSPACE_FILENAME,
    apply_migration,
    audit_workspace,
    init_workspace,
    load_workspace,
    plan_migration,
    plan_template_sync,
    register_workspace,
    utc_now,
)


def _print(payload: Any, *, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload)


def _find_workspace(start: str | Path | None = None) -> Path | None:
    root = Path(start or Path.cwd()).expanduser().resolve()
    if root.is_file():
        root = root.parent
    for candidate in (root, *root.parents):
        manifest = candidate / WORKSPACE_FILENAME
        if manifest.is_file():
            return manifest
    return None


def _source_from_record(version: str, paths: InstallationPaths) -> tuple[Path, Path]:
    root = paths.version_root(version)
    record = load_json_object(root / "installation-record.json")
    source = root / str(dict(record.get("source", {}) or {}).get("path", "source"))
    runtime = dict(record.get("runtime", {}) or {})
    python = Path(str(runtime.get("python") or sys.executable))
    if runtime.get("created"):
        relocated = root / "runtime" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if relocated.is_file():
            python = relocated
    # Preserve the managed virtualenv interpreter path. Resolving its POSIX
    # ``bin/python`` symlink selects the base interpreter and drops the
    # installed OEL dependency environment during command dispatch.
    return source.resolve(), python.expanduser().absolute()


def _selected_engine(paths: InstallationPaths, workspace: Path | None) -> tuple[str | None, Path, Path, str]:
    if workspace is not None:
        loaded = load_workspace(workspace)
        version = str(loaded["engine"]["locked_version"])
        verified = verify_installation(version, paths=paths, full=True)
        if verified["status"] in {"official", "developer"}:
            source, python = _source_from_record(version, paths)
            return version, source, python, verified["status"]
        developer_version = source_project_version() or installed_project_version()
        if developer_version == version:
            source = Path(__file__).resolve().parents[2]
            return version, source, Path(sys.executable), "developer"
        raise RuntimeError(f"Workspace is pinned to OEL {version}, but that version is not installed and verified.")
    current = read_state(paths.current_state, default={}).get("current")
    if current:
        verified = verify_installation(str(current), paths=paths, full=True)
        if verified["status"] in {"official", "developer"}:
            source, python = _source_from_record(str(current), paths)
            return str(current), source, python, verified["status"]
    source = Path(__file__).resolve().parents[2]
    return (
        source_project_version(source_root=source) or installed_project_version(),
        source,
        Path(sys.executable),
        "developer",
    )


@contextmanager
def _engine_lease(paths: InstallationPaths, version: str | None, disposition: str) -> Iterator[None]:
    if not version or disposition == "developer":
        yield
        return
    lease: Path | None = None
    with StateLock(paths.transaction_lock, operation=f"lease:{version}"):
        version_root = paths.version_root(version)
        if not version_root.is_dir():
            raise RuntimeError(f"Selected OEL engine disappeared before it could be leased: {version}")
        lease_root = version_root / "leases"
        lease = lease_root / f"{os.getpid()}-{uuid.uuid4()}.json"
        lease_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            lease,
            {"schema_version": "oel.installation-lease.v1", "pid": os.getpid(), "created_at": utc_now()},
        )
    try:
        yield
    finally:
        assert lease is not None
        try:
            with StateLock(paths.transaction_lock, operation=f"release-lease:{version}"):
                lease.unlink()
        except OSError:
            pass


def _dispatch(command: str, arguments: list[str], *, paths: InstallationPaths, workspace_path: Path | None) -> int:
    version, source, python, disposition = _selected_engine(paths, workspace_path)
    environment = dict(os.environ)
    environment["OEL_ENGINE_VERSION"] = str(version or "unknown")
    environment["OEL_INSTALLATION_DISPOSITION"] = disposition
    environment["OEL_MANAGED_DATA_ROOT"] = str(paths.data_root)
    environment["OEL_MANAGED_CONFIG_ROOT"] = str(paths.config_root)
    if version and paths.version_root(version).is_dir():
        # Managed source trees are content-bound installation evidence. Keep
        # normal Python imports from adding bytecode caches beside that source.
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
    matplotlib_cache = paths.cache / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    if disposition != "developer" and version:
        record = load_json_object(paths.version_root(version) / "installation-record.json")
        environment["OEL_ENGINE_EDITION"] = str(record.get("edition", "unknown"))
        environment["OEL_RELEASE_MANIFEST_SHA256"] = str(record.get("release_manifest_sha256", ""))
        environment["OEL_INSTALLATION_TRANSACTION_ID"] = str(record.get("transaction_id", ""))
    if workspace_path is not None:
        environment["OEL_WORKSPACE_ROOT"] = str(workspace_path.parent)
        workspace = load_workspace(workspace_path)
        environment["OEL_OUTPUT_ROOT"] = str(Path(workspace["root"]) / workspace["paths"]["outputs"])
    else:
        environment["OEL_OUTPUT_ROOT"] = str(paths.data_root / "outputs")
    if command == "trainer":
        argv = [str(python), str(source / "run_game.py"), *arguments]
    elif command in {"sim", "doctor"}:
        argv = [str(python), str(source / "run_simulation.py")]
        argv.extend(["--doctor"] if command == "doctor" else arguments)
    elif command == "twin":
        argv = [str(python), "-m", "sim.spacecraft_twin", *arguments]
    elif command == "review":
        argv = [str(python), "-m", "sim.review", *arguments]
    elif command == "io":
        argv = [str(python), "-m", "sim.data_io", *arguments]
    elif command == "runs":
        argv = [str(python), "-m", "sim.execution.run_lifecycle", *arguments]
    elif command == "study":
        planning_commands = {"schema", "capabilities", "preflight", "plan-review", "prepare-feedback"}
        module = "sim.study_planning" if arguments and arguments[0] in planning_commands else "sim.study"
        argv = [str(python), "-m", module, *arguments]
    elif command == "power":
        argv = [str(python), "-m", "sim.spacecraft_power", *arguments]
    elif command == "lifetime":
        argv = [str(python), "-m", "sim.orbit_lifetime", *arguments]
    elif command == "fsw":
        argv = [str(python), "-m", "sim.fsw_authoring"]
        if workspace_path is not None:
            argv.extend(["--workspace-root", str(workspace_path.parent)])
        argv.extend(arguments)
    elif command == "fswdk":
        argv = [str(python), "-m", "sim.fswdk"]
        if workspace_path is not None:
            argv.extend(["--workspace-root", str(workspace_path.parent)])
        argv.extend(arguments)
    elif command == "mcp":
        argv = [str(python), "-m", "integrations.oel_mcp", *arguments]
    elif command == "hosted":
        argv = [str(python), "-m", "sim.hosted_client", *arguments]
    else:
        raise ValueError(f"Unsupported dispatch command: {command}")
    execution_cwd = workspace_path.parent if workspace_path is not None else Path.cwd()
    if command == "trainer":
        execution_cwd = paths.data_root
        execution_cwd.mkdir(parents=True, exist_ok=True)
    with _engine_lease(paths, version, disposition):
        launch_options = {"creationflags": subprocess.CREATE_NO_WINDOW} if command == "trainer" and os.name == "nt" else {}
        completed = subprocess.run(argv, cwd=execution_cwd, env=environment, check=False, **launch_options)
    return int(completed.returncode)


def _write_workspace_pin(workspace_path: Path, version: str, *, paths: InstallationPaths) -> dict[str, Any]:
    with StateLock(paths.transaction_lock, operation=f"workspace-pin:{version}"):
        workspace = load_workspace(workspace_path)
        manifest_path = Path(workspace["manifest_path"])
        pin_state_path = Path(workspace["root"]) / ".oel" / "pins.json"
        originals = {
            manifest_path: manifest_path.read_text(encoding="utf-8"),
            pin_state_path: pin_state_path.read_text(encoding="utf-8") if pin_state_path.is_file() else None,
            paths.workspaces_state: (
                paths.workspaces_state.read_text(encoding="utf-8") if paths.workspaces_state.is_file() else None
            ),
        }
        raw = yaml.safe_load(originals[manifest_path])
        previous = str(raw["engine"]["locked_version"])
        raw["engine"]["locked_version"] = version
        pin_state = read_state(pin_state_path, default={"schema_version": "oel.workspace-pins.v1", "history": []})
        history = list(pin_state.get("history", []) or [])
        history.append({"previous": previous, "current": version, "changed_at": utc_now()})
        pin_state.update({"current": version, "previous": previous, "history": history[-100:]})
        try:
            atomic_write_text(manifest_path, yaml.safe_dump(raw, sort_keys=False))
            atomic_write_json(pin_state_path, pin_state)
            refreshed = load_workspace(workspace_path)
            registry = read_state(
                paths.workspaces_state,
                default={"schema_version": "oel.workspace-registry.v1", "workspaces": {}},
            )
            items = dict(registry.get("workspaces", {}) or {})
            items[refreshed["workspace_id"]] = {
                "path": refreshed["root"],
                "manifest_sha256": refreshed["manifest_sha256"],
                "locked_version": version,
                "registered_at": utc_now(),
            }
            registry.update({"schema_version": "oel.workspace-registry.v1", "workspaces": items})
            atomic_write_json(paths.workspaces_state, registry)
        except Exception:
            for target, text in originals.items():
                if text is None:
                    if target.exists():
                        target.unlink()
                else:
                    atomic_write_text(target, text)
            raise
    return {"status": "ready", "workspace": workspace["workspace_id"], "current": version, "previous": previous}


def _workspace_rollback(workspace_path: Path, *, paths: InstallationPaths) -> dict[str, Any]:
    workspace = load_workspace(workspace_path)
    pin_state_path = Path(workspace["root"]) / ".oel" / "pins.json"
    state = read_state(pin_state_path)
    previous = str(state.get("previous", "") or "")
    if not previous:
        raise RuntimeError("No previous workspace engine pin is recorded.")
    return _write_workspace_pin(workspace_path, previous, paths=paths)


def _fswdk_available() -> bool:
    """Return whether this installed edition includes the private FSWDK CLI."""

    return importlib.util.find_spec("sim.fswdk") is not None


def _dispatch_commands() -> tuple[str, ...]:
    commands = ["sim", "review", "io", "runs", "study", "power", "lifetime", "fsw", "hosted", "trainer"]
    if _fswdk_available():
        commands.append("fswdk")
    if importlib.util.find_spec("sim.spacecraft_twin") is not None:
        commands.append("twin")
    commands.append("mcp")
    return tuple(commands)


def _installation_editions() -> tuple[str, ...]:
    return ("public", "pro") if pro_installation_available() else ("public",)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install, update, and run Orbital Engineering Lab.")
    parser.add_argument("--data-root", type=Path, help="Override the managed OEL data root.")
    parser.add_argument("--config-root", type=Path, help="Override the managed OEL config root.")
    parser.add_argument("--workspace", type=Path, help="Explicit OEL workspace or workspace manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    installation_editions = _installation_editions()
    for name in _dispatch_commands():
        command = sub.add_parser(name, add_help=False)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("doctor")
    sub.add_parser("version")
    support = sub.add_parser("support-receipt")
    support.add_argument("output", type=Path)

    update = sub.add_parser("update")
    update_sub = update.add_subparsers(dest="update_command", required=True)
    update_sub.add_parser("status").add_argument("--full", action="store_true")
    check = update_sub.add_parser("check")
    check.add_argument("--channel-url")
    check.add_argument("--public-keys", type=Path)
    check.add_argument("--edition", choices=installation_editions, default="public")
    check.add_argument("--channel", choices=("stable", "preview"), default="stable")
    download = update_sub.add_parser("download")
    download.add_argument("manifest_url")
    download.add_argument("--public-keys", type=Path)
    install = update_sub.add_parser("install")
    install.add_argument("manifest", help="Local signed manifest path, or 'latest' for the configured channel.")
    install.add_argument("--public-keys", type=Path)
    install.add_argument("--profile", default=None, help="Dependency profile; defaults to the active installation profile.")
    install.add_argument("--channel-url")
    install.add_argument("--edition", choices=installation_editions, default="public")
    install.add_argument("--channel", choices=("stable", "preview"), default="stable")
    install.add_argument("--developer-unsigned", action="store_true")
    install.add_argument("--license", type=Path)
    install.add_argument("--license-public-keys", type=Path)
    install.add_argument("--no-runtime", action="store_true", help=argparse.SUPPRESS)
    bundle = update_sub.add_parser("install-bundle")
    bundle.add_argument("bundle", type=Path)
    bundle.add_argument("--public-keys", type=Path)
    bundle.add_argument("--profile", default="core")
    bundle.add_argument("--developer-unsigned", action="store_true")
    bundle.add_argument("--license", type=Path)
    bundle.add_argument("--license-public-keys", type=Path)
    bundle.add_argument("--no-runtime", action="store_true", help=argparse.SUPPRESS)
    activate_parser = update_sub.add_parser("activate")
    activate_parser.add_argument("version")
    update_sub.add_parser("rollback")
    cleanup_parser = update_sub.add_parser("cleanup")
    cleanup_parser.add_argument("--apply", action="store_true")
    cleanup_parser.add_argument("--keep", type=int, default=2)
    uninstall_parser = update_sub.add_parser("uninstall")
    uninstall_parser.add_argument("version")
    uninstall_parser.add_argument("--apply", action="store_true")
    recover = update_sub.add_parser("recover-lock")
    recover.add_argument("--stale-after-s", type=float, default=3600.0)
    rotate = update_sub.add_parser("rotate-trusted-keys")
    rotate.add_argument("registry", type=Path)
    rotate.add_argument("--current-keys", type=Path)
    configure = update_sub.add_parser("configure-channel")
    configure.add_argument("channel_url")
    configure.add_argument("--edition", choices=("public", "pro"), default="public")
    configure.add_argument("--channel", choices=("stable", "preview"), default="stable")

    workspace = sub.add_parser("workspace")
    workspace_sub = workspace.add_subparsers(dest="workspace_command", required=True)
    init = workspace_sub.add_parser("init")
    init.add_argument("path", type=Path)
    init.add_argument("--id")
    init.add_argument("--engine-version")
    init.add_argument("--engine-requirement")
    init.add_argument("--quickstart-config", type=Path)
    register = workspace_sub.add_parser("register")
    register.add_argument("path", type=Path)
    status = workspace_sub.add_parser("status")
    status.add_argument("path", type=Path, nargs="?")
    audit = workspace_sub.add_parser("check")
    audit.add_argument("path", type=Path)
    audit.add_argument("--against", required=True)
    audit.add_argument("--release-manifest", type=Path)
    migrate = workspace_sub.add_parser("migrate")
    migrate.add_argument("path", type=Path, nargs="?")
    migrate.add_argument("--to")
    migrate.add_argument("--release-manifest", type=Path)
    migrate.add_argument("--apply-plan", type=Path)
    use = workspace_sub.add_parser("use")
    use.add_argument("path", type=Path)
    use.add_argument("version")
    rollback_parser = workspace_sub.add_parser("rollback")
    rollback_parser.add_argument("path", type=Path)
    template = workspace_sub.add_parser("template-check")
    template.add_argument("path", type=Path)
    template.add_argument("--target-manifest", type=Path, required=True)
    template.add_argument("--template-root", type=Path, required=True)
    return parser


def _paths(args: argparse.Namespace) -> InstallationPaths:
    defaults = InstallationPaths.default()
    return InstallationPaths(
        (args.data_root or defaults.data_root).expanduser().resolve(),
        (args.config_root or defaults.config_root).expanduser().resolve(),
    )


def _split_dispatch_argv(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Leave downstream CLI flags untouched after an OEL dispatch command."""

    value_options = {"--data-root", "--config-root", "--workspace"}
    dispatch = set(_dispatch_commands())
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in value_options:
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in value_options):
            index += 1
            continue
        if token in dispatch:
            return argv[: index + 1], argv[index + 1 :]
        index += 1
    return argv, None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser_argv, dispatch_arguments = _split_dispatch_argv(raw_argv)
    args = parser.parse_args(parser_argv)
    if dispatch_arguments is not None:
        args.arguments = dispatch_arguments
    paths = _paths(args)
    workspace_option = args.workspace
    workspace_path = workspace_option if workspace_option and workspace_option.name == WORKSPACE_FILENAME else (
        workspace_option / WORKSPACE_FILENAME if workspace_option else _find_workspace()
    )
    if args.command == "trainer":
        workspace_path = None
    try:
        if args.command in {*_dispatch_commands(), "doctor"}:
            arguments = list(getattr(args, "arguments", []) or [])
            return _dispatch(args.command, arguments, paths=paths, workspace_path=workspace_path)
        if args.command == "version":
            selected, source, _, disposition = _selected_engine(paths, workspace_path)
            _print({"status": "ready", "version": selected, "source": str(source), "disposition": disposition})
            return 0
        if args.command == "support-receipt":
            _print(write_support_receipt(args.output, paths=paths))
            return 0
        if args.command == "update":
            command = args.update_command
            if command == "status":
                payload = installation_status(paths=paths, full=args.full)
            elif command == "check":
                payload = check_channel(
                    args.channel_url,
                    public_keys=keys_from_path(args.public_keys, paths=paths),
                    edition=args.edition,
                    channel=args.channel,
                    paths=paths,
                )
            elif command == "download":
                payload = download_release(
                    args.manifest_url,
                    paths=paths,
                    public_keys=keys_from_path(args.public_keys, paths=paths),
                )
            elif command == "install":
                if args.profile is None:
                    current = read_state(paths.current_state, default={}).get("current")
                    record = paths.version_root(str(current)) / "installation-record.json"
                    args.profile = load_json_object(record).get("profile", "core") if current and record.is_file() else "core"
                unsigned = bool(args.developer_unsigned)
                license_keys = None
                if args.license_public_keys:
                    from sim.licensing.offline import load_public_keys as load_license_public_keys

                    license_keys = load_license_public_keys(args.license_public_keys)
                release_keys = None if unsigned else keys_from_path(args.public_keys, paths=paths)
                if args.manifest == "latest":
                    if unsigned:
                        raise ValueError("`oel update install latest` requires signed release metadata.")
                    payload = install_latest_release(
                        paths=paths,
                        public_keys=release_keys,
                        channel_url=args.channel_url,
                        edition=args.edition,
                        channel=args.channel,
                        profile=args.profile,
                        create_runtime=not args.no_runtime,
                        license_path=args.license,
                        license_public_keys=license_keys,
                    )
                else:
                    payload = install_release(
                        Path(args.manifest),
                        paths=paths,
                        public_keys=release_keys,
                        require_signature=not unsigned,
                        profile=args.profile,
                        create_runtime=not args.no_runtime,
                        license_path=args.license,
                        license_public_keys=license_keys,
                    )
            elif command == "install-bundle":
                unsigned = bool(args.developer_unsigned)
                license_keys = None
                if args.license_public_keys:
                    from sim.licensing.offline import load_public_keys as load_license_public_keys

                    license_keys = load_license_public_keys(args.license_public_keys)
                payload = install_bundle(
                    args.bundle,
                    paths=paths,
                    public_keys=None if unsigned else keys_from_path(args.public_keys, paths=paths),
                    require_signature=not unsigned,
                    profile=args.profile,
                    create_runtime=not args.no_runtime,
                    license_path=args.license,
                    license_public_keys=license_keys,
                )
            elif command == "activate":
                payload = activate(args.version, paths=paths)
            elif command == "rollback":
                payload = rollback(paths=paths)
            elif command == "cleanup":
                payload = cleanup(paths=paths, dry_run=not args.apply, keep=args.keep)
            elif command == "uninstall":
                payload = uninstall(args.version, paths=paths, dry_run=not args.apply)
            elif command == "recover-lock":
                payload = recover_lock(paths=paths, stale_after_s=args.stale_after_s)
            elif command == "rotate-trusted-keys":
                payload = rotate_trusted_release_keys(
                    args.registry,
                    paths=paths,
                    current_keys=keys_from_path(args.current_keys, paths=paths),
                )
            elif command == "configure-channel":
                payload = configure_channel(
                    args.channel_url,
                    edition=args.edition,
                    channel=args.channel,
                    paths=paths,
                    source="explicit-cli",
                )
            else:
                parser.error(f"Unknown update command: {command}")
                return 2
            _print(payload)
            return 0
        if args.command == "workspace":
            command = args.workspace_command
            if command == "init":
                quickstart = args.quickstart_config
                if quickstart is None:
                    quickstart = quickstart_config_path()
                engine_version = args.engine_version
                if engine_version is None:
                    engine_version, _, _, _ = _selected_engine(paths, None)
                payload = init_workspace(
                    args.path,
                    workspace_id=args.id,
                    engine_version=engine_version,
                    engine_requirement=args.engine_requirement,
                    quickstart_config=quickstart,
                    example_configs=(resource_path("configs", "acceptance_relative_coast.yaml"),),
                )
                register_workspace(args.path, registry_path=paths.workspaces_state, lock_path=paths.transaction_lock)
            elif command == "register":
                payload = register_workspace(
                    args.path, registry_path=paths.workspaces_state, lock_path=paths.transaction_lock
                )
            elif command == "status":
                target = args.path or (workspace_path.parent if workspace_path else None)
                if target is None:
                    payload = read_state(paths.workspaces_state, default={"schema_version": "oel.workspace-registry.v1", "workspaces": {}})
                else:
                    payload = {"status": "ready", "workspace": load_workspace(target)}
            elif command == "check":
                release = load_json_object(args.release_manifest) if args.release_manifest else None
                payload = audit_workspace(args.path, target_version=args.against, release_manifest=release)
            elif command == "migrate":
                if args.apply_plan:
                    payload = apply_migration(args.apply_plan)
                else:
                    if args.path is None or not args.to:
                        raise ValueError("workspace migrate requires PATH and --to, or --apply-plan PLAN.")
                    release = load_json_object(args.release_manifest) if args.release_manifest else None
                    payload = plan_migration(args.path, target_version=args.to, release_manifest=release)
            elif command == "use":
                verified = verify_installation(args.version, paths=paths, full=False)
                if verified["status"] not in {"official", "developer"}:
                    raise RuntimeError(f"Target OEL {args.version} is not installed and verified.")
                report = audit_workspace(args.path, target_version=args.version)
                if report["status"] not in {"compatible", "compatible_with_warnings"}:
                    raise RuntimeError(f"Workspace cannot adopt OEL {args.version}: {report['status']}")
                payload = _write_workspace_pin(args.path, args.version, paths=paths)
            elif command == "rollback":
                payload = _workspace_rollback(args.path, paths=paths)
            elif command == "template-check":
                payload = plan_template_sync(
                    args.path,
                    target_template_manifest=args.target_manifest,
                    template_root=args.template_root,
                )
            else:
                parser.error(f"Unknown workspace command: {command}")
                return 2
            _print(payload)
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        _print({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
