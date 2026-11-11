"""Local exercise entry points. No release or external spacecraft actions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

from .configure_yamcs import MAVEN_POM, configure
from .contracts import Policy
from .endpoint import OperationsEndpoint, serve
from .engineering import SpacecraftPackage
from .temperature_contracts import ThermalProcedurePolicy
from .temperature_source import PlaybackTemperatureSource, SyntheticPayloadDevice
from .yamcs import YamcsClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("configure-yamcs")
    config.add_argument("--root", type=Path, default=Path("build/agent-operations/yamcs"))
    prepare = commands.add_parser("prepare-runtime")
    prepare.add_argument("--root", type=Path, default=Path("build/agent-operations/runtime"))
    yamcs = commands.add_parser("run-yamcs")
    yamcs.add_argument("--root", type=Path, default=Path("build/agent-operations/yamcs"))
    yamcs.add_argument("--java", default="java")
    yamcs.add_argument("--classpath", type=Path, default=Path("build/agent-operations/runtime/classpath.txt"))
    endpoint = commands.add_parser("endpoint")
    endpoint.add_argument("--config", default="configs/agent_operations_pointing.yaml")
    endpoint.add_argument("--spacecraft-id", default="oel-sat-a")
    endpoint.add_argument("--evidence-dir", type=Path, default=Path("build/agent-operations/evidence"))
    endpoint.add_argument(
        "--temperature-fixture", type=Path, help="Explicit synthetic observation playback; never a thermal model"
    )
    endpoint.add_argument("--temperature-policy", type=Path, help="Trusted owner-configured procedure limits JSON")
    for sub in (endpoint,):
        sub.add_argument("--credentials", type=Path, default=Path("build/agent-operations/yamcs/credentials.json"))
        sub.add_argument("--yamcs-url", default="http://127.0.0.1:18090")
    args = parser.parse_args()
    if args.command == "prepare-runtime":
        args.root.mkdir(parents=True, exist_ok=True)
        (args.root / "pom.xml").write_text(MAVEN_POM)
        print(f"Prepared pinned Yamcs runtime Maven project in {args.root}")
        return
    if args.command == "run-yamcs":
        # Resolve before changing directory; no shell, credentials, or model input.
        classpath = args.classpath.resolve().read_text().strip()
        command = [
            str(Path(args.java).resolve()) if Path(args.java).exists() else args.java,
            "-Xmx256m",
            "-XX:ActiveProcessorCount=2",
            "-Dio.netty.eventLoopThreads=2",
            "-cp",
            classpath,
            "org.yamcs.YamcsServer",
            "--log",
            "0",
        ]
        try:
            subprocess.run(command, cwd=args.root.resolve(), check=True)
        except KeyboardInterrupt:
            pass
        return
    if args.command == "configure-yamcs":
        path = configure(args.root)
        print(f"Configured local Yamcs in {args.root}; credentials saved to {path}")
        return
    credentials = json.loads(args.credentials.read_text())
    reader = YamcsClient(args.yamcs_url, "endpoint", credentials["endpoint"])
    if args.command == "endpoint":
        package = SpacecraftPackage(yaml.safe_load(Path(args.config).read_text()), args.spacecraft_id)
        temperature = {}
        if args.temperature_fixture:
            procedure_policy = (
                ThermalProcedurePolicy(**json.loads(args.temperature_policy.read_text()))
                if args.temperature_policy
                else ThermalProcedurePolicy()
            )
            temperature = {
                "temperature_source": PlaybackTemperatureSource(args.temperature_fixture, procedure_policy),
                "payload_device": SyntheticPayloadDevice(),
                "temperature_policy": procedure_policy,
            }
        elif args.temperature_policy:
            parser.error("--temperature-policy requires --temperature-fixture for this local exercise")
        endpoint = OperationsEndpoint(package, Policy(), reader, args.evidence_dir, **temperature)
        server = serve(endpoint)
        print(
            json.dumps({"status": "ready", "endpoint": "http://127.0.0.1:18091", "package": package.describe()}),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            endpoint.close()


if __name__ == "__main__":
    main()
