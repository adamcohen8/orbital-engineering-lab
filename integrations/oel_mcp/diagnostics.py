from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from integrations.oel_mcp.execution import ExecutionApprovalPolicy
from integrations.oel_mcp.policy import MCPPathPolicy
from integrations.oel_mcp.public_registry import public_contracts_for_profile
from integrations.oel_mcp.sdk_protocol import MCP_SDK_PROTOCOL_VERSION, MCP_SDK_REQUIREMENT, installed_sdk_version


def doctor_report(*, profile: str, adapter: str) -> dict[str, Any]:
    return _doctor_report_base(
        profile=profile,
        adapter=adapter,
        contracts=public_contracts_for_profile(profile),
        entitlement_rows=[],
    )


def _doctor_report_base(
    *,
    profile: str,
    adapter: str,
    contracts: tuple[Any, ...],
    entitlement_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path_policy = MCPPathPolicy.configured()
    approval_policy = ExecutionApprovalPolicy.configured()
    grant_counts = _grant_counts(approval_policy)
    sdk_version = installed_sdk_version()
    launch_command, launch_args, launch_source = default_host_launch()
    source_version = _source_project_version()
    distribution_version = _installed_version("orbital-engineering-lab")
    checks: list[dict[str, Any]] = []
    _check(checks, "adapter", adapter in {"sdk", "legacy"}, detail={"selected": adapter})
    _check(
        checks,
        "sdk_dependency",
        adapter == "legacy" or _supported_sdk_version(sdk_version),
        required=adapter == "sdk",
        detail={"installed": sdk_version, "requirement": MCP_SDK_REQUIREMENT},
    )
    _check(
        checks,
        "read_roots",
        bool(path_policy.read_roots) and all(path.is_dir() for path in path_policy.read_roots),
        detail={"count": len(path_policy.read_roots), "roots": [str(path) for path in path_policy.read_roots]},
    )
    _check(
        checks,
        "write_roots",
        bool(path_policy.write_roots) and all(_writable_root(path) for path in path_policy.write_roots),
        required=False,
        detail={"count": len(path_policy.write_roots), "roots": [str(path) for path in path_policy.write_roots]},
    )
    _check(
        checks,
        "registry",
        bool(contracts),
        detail={
            "profile": profile,
            "tool_count": len(contracts),
            "read_tools": sum(not item.writes and not item.executes for item in contracts),
            "write_tools": sum(item.writes and not item.executes for item in contracts),
            "execute_tools": sum(item.executes for item in contracts),
        },
    )
    _check(
        checks,
        "host_launch",
        Path(launch_command).is_file() or shutil.which(launch_command) is not None,
        detail={"command": launch_command, "args": launch_args, "source": launch_source},
    )
    _check(
        checks,
        "operator_approvals",
        bool(
            approval_policy.write_approval_ids
            or approval_policy.execution_approval_ids
            or approval_policy.trust_approval_ids
            or grant_counts["active_bounded_grants"]
        ),
        required=False,
        detail={
            "write_ids": len(approval_policy.write_approval_ids),
            "execute_ids": len(approval_policy.execution_approval_ids),
            "trust_ids": len(approval_policy.trust_approval_ids),
            **grant_counts,
            "meaning": "Counts describe configuration only; every call checks scope, input binding, expiry and revocation.",
        },
    )
    if grant_counts["active_bounded_effect_grants"]:
        _check(
            checks,
            "bounded_operation_ledger",
            bool(os.environ.get("OEL_MCP_OPERATION_LEDGER_ROOT", "").strip()),
            required=False,
            detail={
                "configured": bool(os.environ.get("OEL_MCP_OPERATION_LEDGER_ROOT", "").strip()),
                "meaning": "Bounded effects require a valid host-private durable ledger outside authorized data roots; calls verify its location.",
            },
        )
    if entitlement_rows:
        _check(
            checks,
            "pro_entitlements",
            all(row["available"] for row in entitlement_rows),
            required=False,
            detail={"features": entitlement_rows},
        )
    failed = [row for row in checks if row["required"] and not row["passed"]]
    warnings = [row for row in checks if not row["required"] and not row["passed"]]
    return {
        "schema_version": 1,
        "status": "failed" if failed else "ready_with_disabled_effects" if warnings else "ready",
        "python_version": platform.python_version(),
        "oel_version": source_version or distribution_version,
        "oel_version_source": "source_pyproject" if source_version else "installed_distribution",
        "installed_distribution_version": distribution_version,
        "mcp_sdk_version": sdk_version,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "profile": profile,
        "adapter": adapter,
        "transport": "stdio",
        "network_listener": False,
        "checks": checks,
        "next_steps": _next_steps(failed=failed, warnings=warnings),
    }


def _grant_counts(policy: ExecutionApprovalPolicy) -> dict[str, int]:
    """Report grant readiness without revealing IDs or reviewed request data."""
    store = policy.grant_store
    grants = () if store is None else store.grants
    revoked = frozenset() if store is None else store.revoked_ids
    now = datetime.now(timezone.utc)
    active = [
        grant for grant in grants
        if grant.bounded and not grant.revoked and grant.grant_id not in revoked
        and (grant.expires_at is None or now < grant.expires_at.astimezone(timezone.utc))
    ]
    return {
        "bounded_grants": sum(grant.bounded for grant in grants),
        "active_bounded_grants": len(active),
        "active_bounded_effect_grants": sum(bool(grant.scopes & {"write", "execute"}) for grant in active),
    }


def host_config(
    *,
    host: str,
    command: str,
    command_args: tuple[str, ...] = (),
    cwd: Path,
    profile: str,
) -> str:
    env = {
        "OEL_MCP_ADAPTER": "sdk",
        "OEL_MCP_READ_ROOTS": str(cwd),
        "OEL_MCP_WRITE_ROOTS": str(cwd / "outputs"),
    }
    if profile != "public_local":
        env["OEL_MCP_PROFILE"] = profile
    if host == "claude":
        return json.dumps(
            {
                "mcpServers": {
                    "oel": {
                        "type": "stdio",
                        "command": command,
                        "args": list(command_args),
                        "cwd": str(cwd),
                        "env": env,
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
    if host == "codex":
        lines = [
            "[mcp_servers.oel]",
            f"command = {json.dumps(command)}",
            f"args = {json.dumps(list(command_args), separators=(',', ':'))}",
            f"cwd = {json.dumps(str(cwd))}",
            "required = true",
            "",
            "[mcp_servers.oel.env]",
        ]
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in sorted(env.items()))
        return "\n".join(lines)
    raise ValueError("Supported host config targets are codex and claude.")


def default_host_launch(profile: str = "public_local") -> tuple[str, tuple[str, ...], str]:
    if str(profile).strip() not in {"public_local", "direct_frontier_restricted"}:
        return (
            str(Path(sys.executable).absolute()),
            ("-m", "integrations.oel_mcp.server"),
            "private_python_module_entrypoint",
        )
    entrypoint = shutil.which("oel-mcp")
    if entrypoint:
        return str(Path(entrypoint).absolute()), (), "installed_console_entrypoint"
    return str(Path(sys.executable).absolute()), ("-m", "integrations.oel_mcp"), "python_module_fallback"


def run_server_cli(
    *,
    argv: list[str] | None,
    default_profile: str,
    serve: Callable[[str], None],
    doctor: Callable[[str, str], dict[str, Any]] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run or diagnose the supported local OEL MCP stdio server.")
    parser.add_argument("--doctor", action="store_true", help="Check local MCP readiness without starting a server.")
    parser.add_argument("--print-host-config", choices=("codex", "claude"))
    parser.add_argument("--command", help="Command used in generated host configuration.")
    parser.add_argument(
        "--arg",
        dest="command_args",
        action="append",
        default=[],
        help="Repeat for each argument passed to the generated MCP server command.",
    )
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--profile", default=os.environ.get("OEL_MCP_PROFILE", default_profile))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    adapter = os.environ.get("OEL_MCP_ADAPTER", "sdk").strip().lower()
    if args.doctor:
        report = (doctor or _default_doctor)(str(args.profile), adapter)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1 if report["status"] == "failed" else 0
    if args.print_host_config:
        if args.command:
            command = str(args.command)
            command_args = tuple(str(item) for item in args.command_args)
            if not command_args and Path(command).name.lower().startswith("python"):
                command_args = (
                    "-m",
                    "integrations.oel_mcp"
                    if str(args.profile) in {"public_local", "direct_frontier_restricted"}
                    else "integrations.oel_mcp.server",
                )
        else:
            command, command_args, _source = default_host_launch(str(args.profile))
        print(
            host_config(
                host=args.print_host_config,
                command=command,
                command_args=command_args,
                cwd=args.cwd.expanduser().resolve(),
                profile=str(args.profile),
            )
        )
        return 0
    serve(str(args.profile))
    return 0


def _supported_sdk_version(value: str | None) -> bool:
    if not value:
        return False
    try:
        major = int(value.split(".", 1)[0])
    except ValueError:
        return False
    return major == 2


def _default_doctor(profile: str, adapter: str) -> dict[str, Any]:
    return doctor_report(profile=profile, adapter=adapter)


def _writable_root(path: Path) -> bool:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _source_project_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        in_project = False
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                in_project = stripped == "[project]"
                continue
            if in_project and stripped.startswith("version"):
                value = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                return value or None
    except OSError:
        return None
    return None


def _check(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    required: bool = True,
    detail: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {"check_id": check_id, "passed": bool(passed), "required": bool(required), "detail": dict(detail or {})}
    )


def _next_steps(*, failed: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    if any(row["check_id"] == "sdk_dependency" for row in failed):
        steps.append('Install the bounded MCP profile with `python -m pip install ".[mcp]"`.')
    if any(row["check_id"] in {"read_roots", "write_roots"} for row in (*failed, *warnings)):
        steps.append("Configure narrow existing OEL_MCP_READ_ROOTS and OEL_MCP_WRITE_ROOTS.")
    if any(row["check_id"] == "operator_approvals" for row in warnings):
        steps.append("Configure purpose-specific approval IDs only when write, execute, or plugin trust is required.")
    if any(row["check_id"] == "bounded_operation_ledger" for row in warnings):
        steps.append("Configure a host-private OEL_MCP_OPERATION_LEDGER_ROOT outside authorized data roots for bounded effects.")
    if any(row["check_id"] == "pro_entitlements" for row in warnings):
        steps.append("Install and validate the required offline Pro license before using entitled tools.")
    return steps


__all__ = ["default_host_launch", "doctor_report", "host_config", "run_server_cli"]
