"""Agent-friendly CLI. Credential setup is local and deliberately interactive."""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from .contracts import AcquisitionError, digest, strict_json
from .files import publish, read_snapshot
from .providers import describe_providers
from .service import AcquisitionService


def document(path):
    return strict_json(read_snapshot(Path(path).expanduser().absolute(), 2 * 1024 * 1024))


def parser():
    root = argparse.ArgumentParser(prog="oel data", description="Bounded local data acquisition; no simulation execution.")
    root.add_argument("--offline", action="store_true")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("providers")
    sub = commands.add_parser("schema")
    sub.add_argument("kind", choices=["request", "receipt"])
    for name in ("plan", "fetch"):
        sub = commands.add_parser(name)
        sub.add_argument("request")
    sub = commands.add_parser("register")
    sub.add_argument("path")
    sub.add_argument("--kind", required=True, choices=["gravity", "catalog", "catalog-metadata", "eop", "space-weather", "opaque"])
    sub.add_argument("--format", required=True)
    sub.add_argument("--terms")
    sub.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    for name in ("inspect", "check", "materialize"):
        sub = commands.add_parser(name)
        sub.add_argument("artifact")
        if name == "check":
            sub.add_argument("--requirements", required=True)
        if name == "materialize":
            sub.add_argument("--destination", required=True)
            sub.add_argument("--requirements")
    sub = commands.add_parser("select")
    sub.add_argument("artifacts", nargs="+")
    sub.add_argument("--object-ids", type=int, nargs="+", required=True)
    sub.add_argument("--at", required=True)
    sub.add_argument("--policy", choices=["exact", "latest-before"], default="latest-before")
    sub.add_argument("--max-age-s", type=float, default=86400)
    sub.add_argument("--published-before")
    sub = commands.add_parser("build-catalog-study")
    sub.add_argument("artifact")
    sub.add_argument("--scenario-path", required=True)
    sub.add_argument("--scenario-name", required=True)
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--start", required=True)
    sub.add_argument("--duration-s", type=float, required=True)
    sub.add_argument("--step-s", type=float, required=True)
    sub = commands.add_parser("bind")
    sub.add_argument("scenario")
    sub.add_argument("--input", required=True)
    sub.add_argument("--field", required=True)
    sub.add_argument("--out", required=True)
    commands.add_parser("retention")
    sub = commands.add_parser("connections").add_subparsers(dest="operation", required=True)
    for name in ("add", "status", "test", "remove"):
        command = sub.add_parser(name)
        if name == "add":
            command.add_argument("provider", choices=["spacetrack"])
            command.add_argument("--name", required=True)
        else:
            command.add_argument("name")
    return root


def dispatch(args, service):
    if args.command == "providers":
        return describe_providers()
    if args.command == "schema":
        from importlib.resources import files

        return strict_json(files("sim.data_acquisition").joinpath("schemas", f"{args.kind}-v1.schema.json").read_bytes())
    if args.command == "connections":
        if args.operation == "add":
            if not sys.stdin.isatty():
                raise AcquisitionError("INTERACTIVE_SETUP_REQUIRED", "Run connection setup in your local terminal; credentials are not accepted through arguments or agent input.")
            identity = getpass.getpass("Space-Track login identity: ")
            password = getpass.getpass("Space-Track password: ")
            return service.connections.add(args.name, identity, password)
        if args.operation == "test":
            return service.test_connection(args.name, offline=args.offline)
        return getattr(service.connections, args.operation)(args.name)
    if args.command == "plan":
        return service.plan(document(args.request))
    if args.command == "fetch":
        return service.fetch(document(args.request), offline=args.offline)
    if args.command == "register":
        return service.register(args.path, kind=args.kind, format=args.format, terms=document(args.terms) if args.terms else None, max_bytes=args.max_bytes)
    if args.command == "inspect":
        return service.inspect(args.artifact)
    if args.command == "check":
        return service.check(args.artifact, document(args.requirements))
    if args.command == "materialize":
        return service.materialize(args.artifact, args.destination, requirements=document(args.requirements) if args.requirements else None)
    if args.command == "select":
        return service.select(args.artifacts, object_ids=args.object_ids, at=args.at, policy=args.policy,
                              max_age_s=args.max_age_s, published_before=args.published_before)
    if args.command == "build-catalog-study":
        return service.build_catalog_study(args.artifact, scenario_path=args.scenario_path, scenario_name=args.scenario_name,
            output_dir=args.output_dir, start=args.start, duration_s=args.duration_s, step_s=args.step_s)
    if args.command == "bind":
        # Parse YAML only; do not import plugins from an unfamiliar scenario.
        import yaml

        from sim.api import SimulationWorkspace
        from sim.config.scenario.loader import _parse_yaml_mapping
        from sim.security import ConfigPathPolicy

        from .binding import bind_file
        source, input_path, destination = (Path(p).expanduser().absolute() for p in (args.scenario, args.input, args.out))
        config = _parse_yaml_mapping(read_snapshot(source, 2 * 1024 * 1024), source)
        policy = ConfigPathPolicy.default(config_path=source, read_roots=(input_path.parent,))
        result = bind_file(config, path=input_path, field=args.field, path_policy=policy)
        payload = yaml.safe_dump(result, sort_keys=False).encode()
        publish(destination, payload)
        try:
            report = SimulationWorkspace(read_roots=(source.parent, input_path.parent)).validate_safe(destination)
            if not report.get("ok", False):
                raise AcquisitionError("INVALID_SCENARIO", "Bound scenario failed safe structural validation.")
        except Exception:
            # Remove only this command's unchanged, newly created draft.
            if digest(read_snapshot(destination, 2 * 1024 * 1024)) == digest(payload):
                destination.unlink()
            raise
        return {"scenario_path": str(destination), "execution_authorized": False}
    return service.store.retention()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        value = dispatch(args, AcquisitionService())
        print(json.dumps({"ok": True, "result": value}, indent=2, allow_nan=False))
        return 0
    except AcquisitionError as exc:
        print(json.dumps(exc.result(), allow_nan=False))
        return 2
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        # Avoid echoing source contents, endpoints or local credentials in tracebacks.
        print(json.dumps({"ok": False, "error": "INVALID_INPUT", "message": "Input could not be read or did not satisfy its documented contract."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
